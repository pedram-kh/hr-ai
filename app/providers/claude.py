"""Claude answer-synthesis adapter (ADR-0015, default provider).

Builds the constrained synthesis prompt (a module constant — never assembled in
business logic), calls the Anthropic Messages API with the per-call key, parses
the structured JSON envelope, and maps the model's cited source indices back to
the real chunk ids / pages / authority levels from the input set.

Two safety properties live here:
- The prompt encodes the AUTHORITY-PRECEDENCE rule explicitly (convenio governs
  where it speaks; the Estatuto/national_law is only the gap-filling baseline;
  never blend; never silently present the baseline as the answer).
- Citations are mapped from the provided set only — a cited index outside the set
  is dropped, so a hallucinated citation can never reach hr-backend. `grounded`
  is set false when nothing valid was cited.
"""

from __future__ import annotations

import json
import re
import time

from .base import (
    AnswerProvider,
    ChunkInput,
    GroundChunk,
    GroundingResult,
    ProviderConfig,
    RouterResult,
    SynthesisResult,
)

# Authority ordering for the precedence rule. Lower index = higher precedence for
# the topics it addresses. `official_convenio` and `internal_hr_ruling` are the
# employee's specific agreement; `national_law` (the Estatuto) is the baseline.
_AUTHORITY_RANK = {
    "internal_hr_ruling": 0,
    "official_convenio": 0,
    "national_law": 1,
}

SYSTEM_PROMPT = (
    "Eres un asistente de Recursos Humanos especializado en derecho laboral "
    "español y en convenios colectivos. Respondes a personas trabajadoras sobre "
    "las condiciones que les aplican.\n\n"
    "REGLAS ABSOLUTAS:\n"
    "1. Afirma un dato SOLO si está respaldado DIRECTA y LITERALMENTE por una de las "
    "FUENTES proporcionadas. PROHIBIDO usar tu conocimiento general del derecho "
    "laboral español para rellenar huecos, completar cifras, o 'recordar' lo que dice "
    "una ley. Si ninguna fuente contiene el dato, ese dato NO puede aparecer en la "
    "respuesta. (Ejemplo de error grave a evitar: inventar plazos de periodo de prueba "
    "de memoria cuando ninguna fuente los enuncia.)\n"
    "2. CITA OBLIGATORIA POR AFIRMACIÓN: CADA afirmación sustantiva (toda regla, cifra, "
    "derecho, duración, condición o ámbito) debe llevar su PROPIO marcador [Fuente N] "
    "apuntando a la fuente que REALMENTE enuncia ESE dato — nunca a la fuente más "
    "cercana o más parecida. No basta con citar al principio o al final: cada frase con "
    "contenido sustantivo lleva su cita. Si una fuente habla de un tema distinto (p. ej. "
    "la duración de un contrato formativo) NO sirve para responder sobre otro tema "
    "(p. ej. el periodo de prueba), aunque ambos mencionen 'meses'. Y si NINGUNA fuente "
    "proporcionada respalda una afirmación, OMÍTELA por completo: no enuncies contenido "
    "sustantivo que no puedas citar (mejor una respuesta más corta y 100% citada que "
    "una frase sin cita). Un dato sustantivo sin su [Fuente N] es un error.\n"
    "3. REGLA DE PRECEDENCIA: el CONVENIO de la persona trabajadora gobierna los temas "
    "que regula. La LEY NACIONAL (el Estatuto de los Trabajadores) es solo la base "
    "mínima que aplica donde el convenio guarda silencio. Cuando el convenio se "
    "pronuncia sobre el tema preguntado, su respuesta PREVALECE; usa el Estatuto solo "
    "para rellenar lo que el convenio no regula. NUNCA afirmes una regla del convenio "
    "que las fuentes del convenio no contienen.\n"
    "4. Conflicto convenio vs Estatuto sobre la MISMA pregunta: la respuesta es el "
    "dato del CONVENIO (cítalo a su fuente). Puedes añadir el mínimo legal del Estatuto "
    "solo como dato propio adicional y citado a SU fuente, nunca fundido en una sola "
    "cifra; si no puedes determinar con seguridad cuál gobierna, indícalo y pon "
    "confianza baja. NUNCA presentes la base nacional como si fuera la respuesta cuando "
    "el convenio regula el tema.\n"
    "5. PROCEDENCIA vs CONTENIDO: enuncia el CONTENIDO sustantivo (la regla, la cifra, "
    "el derecho, la duración, la condición o el ámbito) y deja que el marcador "
    "[Fuente N] indique de dónde sale. NO escribas frases cuya ÚNICA función sea decir "
    "de qué documento procede un dato (p. ej. «esto está en tu convenio», «según el "
    "Estatuto…», «tu convenio establece que…»): la procedencia ya la transmiten las "
    "citas [Fuente N] y las insignias de autoridad. La PRECEDENCIA (el convenio "
    "gobierna / el Estatuto es la base) se expresa citando cada dato a la fuente "
    "correcta, NO con una frase de procedencia. Esto NO reduce las citas: sigue citando "
    "SIEMPRE cada dato sustantivo con [Fuente N]; solo elimina las frases meta sobre el "
    "origen.\n"
    "6. CUANDO LAS FUENTES NO RESPONDEN la pregunta: \n"
    "   - Si una fuente de LEY NACIONAL (Estatuto) sí la responde, responde a partir de "
    "ella (citándola).\n"
    "   - Si NINGUNA fuente la responde, NO inventes y NO respondas: devuelve "
    "`\"cited_sources\": []` y `\"confidence\"` baja (≤ 0.2), con un `answer` breve que "
    "diga que no dispones de información suficiente. Es PREFERIBLE abstenerse a dar una "
    "respuesta no fundamentada — la abstención hace que el caso se derive a una persona.\n"
    "7. Responde en el MISMO idioma que la pregunta.\n\n"
    "FORMATO DE SALIDA: devuelve EXCLUSIVAMENTE un objeto JSON válido, sin texto "
    "alrededor, con esta forma:\n"
    '{"answer": "<respuesta con marcadores [Fuente N], o nota de abstención>", '
    '"cited_sources": [<números N de las fuentes que REALMENTE enuncian lo afirmado; '
    'vacío si te abstienes>], '
    '"confidence": <número entre 0 y 1>}'
)


def _authority_label(level: str | None) -> str:
    return {
        "national_law": "ley nacional / Estatuto (base mínima)",
        "official_convenio": "convenio (gobierna su materia)",
        "internal_hr_ruling": "resolución interna de RR. HH.",
    }.get(level or "", "fuente")


def _build_user_prompt(question: str, chunks: list[ChunkInput]) -> str:
    lines = [f"Pregunta: {question}", "", "FUENTES disponibles:"]
    for idx, c in enumerate(chunks, start=1):
        pages = ""
        if c.page_from is not None:
            pages = f", p. {c.page_from}"
            if c.page_to is not None and c.page_to != c.page_from:
                pages += f"–{c.page_to}"
        lines.append(f"[Fuente {idx}] ({_authority_label(c.authority_level)}{pages}): {c.content}")
    lines.append("")
    lines.append(
        "Recuerda: prioriza el convenio donde regule el tema; usa la ley nacional solo "
        "para lo que el convenio no cubra. Enuncia el contenido sustantivo y deja que "
        "[Fuente N] indique la procedencia — no escribas frases sobre de qué documento "
        "procede un dato. CADA afirmación sustantiva lleva su propio [Fuente N]; si "
        "ninguna fuente la respalda, OMÍTELA (no enuncies datos que no puedas citar). "
        "Responde en el idioma de la pregunta y devuelve solo el JSON."
    )
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """Parse the model's JSON envelope, tolerating ```json fences or stray prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    return json.loads(text)


def _renumber_markers(answer: str, orig_to_display: dict[int, int]) -> str:
    """Rewrite [Fuente N] markers from the model's input-chunk indices to the
    compact display numbers (1..M) of the cited subset (Sprint 2b-2 §7). A marker
    whose index was never mapped (cited an out-of-set / dropped index) is removed
    so the displayed text never references a source not in the FUENTES list."""

    def repl(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        display = orig_to_display.get(idx)
        return f"[Fuente {display}]" if display is not None else ""

    out = re.sub(r"\[Fuente\s+(\d+)\]", repl, answer)
    # Tidy any double spaces / spaces-before-punctuation left by a removed marker.
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([.,;:])", r"\1", out)
    return out.strip()


# --- Router (ADR-0016) — small/fast model classification + decomposition --------
ROUTER_SYSTEM_PROMPT = (
    "Eres un clasificador de preguntas para un asistente de Recursos Humanos "
    "especializado en convenios colectivos españoles. Clasifica CADA pregunta en "
    "una sola etiqueta:\n"
    "- \"salary\": pide una CIFRA de retribución/salario/sueldo/nómina/tablas "
    "salariales/pagas/€ por hora (cuánto se cobra/gana en una categoría).\n"
    "- \"prose\": cualquier otra duda sobre condiciones laborales del convenio o la "
    "ley (jornada, vacaciones, permisos, periodo de prueba, excedencias, etc.).\n"
    "- \"off_domain\": no es una cuestión de RR. HH./laboral (p. ej. cocina, "
    "deportes, política, fiscalidad personal).\n\n"
    "ADEMÁS, si la pregunta es COMPUESTA (contiene DOS O MÁS subpreguntas o temas "
    "distintos, normalmente unidos por 'y'/'además'/comas o varios signos de "
    "interrogación), descomponla en una lista de subpreguntas autónomas, una por "
    "tema, reformulada para buscarse por separado. Si es de un solo tema, devuelve "
    "subqueries vacío. La etiqueta de una pregunta compuesta es la del tema "
    "predominante (normalmente \"prose\").\n\n"
    "FORMATO: devuelve EXCLUSIVAMENTE un objeto JSON válido, sin texto alrededor:\n"
    '{"label": "salary|prose|off_domain", '
    '"confidence": <número entre 0 y 1>, '
    '"subqueries": [<subpreguntas autónomas, o vacío si es de un solo tema>]}'
)


# --- Grounding (Sprint 2b-2 §5) — per-claim entailment, capable model ----------
GROUND_SYSTEM_PROMPT = (
    "Eres un verificador de fundamentación (grounding) para respuestas de un "
    "asistente de RR. HH. Recibes una PREGUNTA, una RESPUESTA propuesta, y las "
    "FUENTES que la respuesta citó. Tu tarea: descomponer la respuesta en "
    "afirmaciones atómicas, CLASIFICAR cada una y, si procede, decidir si está "
    "ENTRAÑADA (directamente respaldada) por alguna de las FUENTES citadas.\n\n"
    "CLASIFICA cada afirmación con \"tipo\":\n"
    "- \"sustantiva\": enuncia CONTENIDO de la respuesta — una regla, una cifra, un "
    "derecho, una duración, una condición o un ámbito. DEBE estar ENTRAÑADA por "
    "alguna FUENTE citada.\n"
    "- \"procedencia\": SOLO indica de qué documento sale la respuesta, sin aportar "
    "contenido (p. ej. «esto está en tu convenio», «según el Estatuto», «tu convenio "
    "establece que…»). La procedencia ya la transmite la cita, así que NO se somete a "
    "entrañamiento: márcala grounded=true y supporting_source=null.\n\n"
    "GUÍA DE PRECISIÓN (CRÍTICA — no la relajes): la exención de «procedencia» es "
    "SOLO para frases SIN contenido sustantivo. Si una frase con envoltorio atributivo "
    "contiene un dato, es SUSTANTIVA y debe entrañarse: «tu convenio te da 31 días» → "
    "la parte «31 días» es sustantiva; «según el Estatuto, el preaviso es de 15 días» → "
    "«preaviso de 15 días» es sustantiva. Solo «esto está en tu convenio» (sin cifra ni "
    "regla) es procedencia pura. ANTE LA DUDA, clasifícala como SUSTANTIVA. JAMÁS dejes "
    "escapar una cifra, regla, derecho, duración, condición o ámbito inventado "
    "etiquetándolo «procedencia».\n\n"
    "REGLAS ESTRICTAS (para las afirmaciones sustantivas):\n"
    "1. Entrañamiento = la fuente AFIRMA el contenido de la afirmación, no solo "
    "que aparezca una palabra o un número suelto. La mera presencia de un dígito o "
    "término NO es fundamentación.\n"
    "2. CUIDADO CON TABLAS: si una fuente está marcada como tabular/columnas, que "
    "un número aparezca en ella NO entraña una afirmación sobre ese número salvo "
    "que la fila/columna/contexto lo respalde inequívocamente. Ante la duda con "
    "datos tabulares, NO está fundamentada.\n"
    "3. Sé estricto: una afirmación sustantiva que la fuente no respalda "
    "DIRECTAMENTE es no_fundamentada, aunque sea plausible o de conocimiento general.\n"
    "4. Ignora cortesías y conectores sin contenido factual (no son afirmaciones).\n\n"
    "FORMATO: devuelve EXCLUSIVAMENTE un objeto JSON válido:\n"
    '{"claims": [{"claim": "<afirmación>", "tipo": "sustantiva|procedencia", '
    '"grounded": <true|false>, '
    '"supporting_source": <número de FUENTE que la respalda, o null>}], '
    '"all_grounded": <true si TODAS las afirmaciones SUSTANTIVAS están fundamentadas, '
    "si no false>}"
)

# Output-token budget for the per-claim grounding JSON (Sprint 2b-2 Correction-04).
# A rich multi-claim answer (e.g. a per-article vacaciones answer after the 2c
# re-chunk) needs ~1,150 tokens of claim-by-claim JSON; the original 1024 cap
# truncated the response (stop_reason="max_tokens"), `_extract_json` then failed,
# and the conservative "unparseable → not grounded" branch escalated a
# FULLY-grounded answer (the terse-vacaciones residual). 4096 gives comfortable
# headroom (worst observed ground response ≈1,152 tok). On the rare residual
# truncation we retry ONCE at a larger budget before escalating — and only with a
# DISTINCT `grounding_truncated` trace note, never silently conflated with a
# genuine ungrounded claim. This gives the gate room to finish; it never weakens
# it (the "unparseable/truncated → escalate" floor stays).
GROUND_MAX_TOKENS = 4096
GROUND_MAX_TOKENS_RETRY = 8192


def _build_ground_prompt(question: str, answer: str, chunks: list[GroundChunk]) -> str:
    lines = [f"Pregunta: {question}", "", f"Respuesta propuesta:\n{answer}", "", "FUENTES citadas:"]
    for idx, c in enumerate(chunks, start=1):
        tag = " [TABLA/COLUMNAS]" if c.is_tabular else ""
        lines.append(f"[Fuente {idx}]{tag}: {c.content}")
    lines.append("")
    lines.append(
        "Clasifica cada afirmación (sustantiva/procedencia) y evalúa el entrañamiento "
        "SOLO de las sustantivas. Recuerda la guía de precisión: una cifra/regla con "
        "envoltorio atributivo es sustantiva. Devuelve solo el JSON."
    )
    return "\n".join(lines)


class ClaudeProvider(AnswerProvider):
    def synthesise(
        self,
        question: str,
        chunks: list[ChunkInput],
        api_key: str,
        config: ProviderConfig,
    ) -> SynthesisResult:
        import anthropic  # imported lazily so the dep is only needed at call time

        client = anthropic.Anthropic(api_key=api_key, base_url=config.endpoint or None)
        user_prompt = _build_user_prompt(question, chunks)

        started = time.monotonic()
        resp = client.messages.create(
            model=config.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)

        raw_text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

        try:
            envelope = _extract_json(raw_text)
        except (json.JSONDecodeError, ValueError):
            # Unparseable output → ungrounded. hr-backend escalates (no guess).
            return SynthesisResult(
                answer="",
                citations=[],
                grounding_signal={"grounded": False, "citation_count": 0, "top_chunk_score": 0.0},
                confidence=0.0,
                authority_used=[],
                trace_fragment={
                    "provider": config.provider,
                    "model": config.model,
                    "synthesis_ms": elapsed_ms,
                    "parse_error": True,
                },
            )

        answer = str(envelope.get("answer", "")).strip()
        confidence = float(envelope.get("confidence", 0.0) or 0.0)
        cited_indices = envelope.get("cited_sources", []) or []

        # Map cited 1-based indices back to the REAL chunks. Drop any index outside
        # the provided set (a hallucinated citation can never pass through).
        #
        # Citation-marker numbering (Sprint 2b-2 §7): the model numbers [Fuente N]
        # by the INPUT chunk order (e.g. up to 8), but only the CITED subset is
        # displayed. Left as-is the text could read "[Fuente 4]" with two sources
        # shown. So we renumber the cited subset to a compact 1..M (in citation
        # order) and rewrite the in-text markers, guaranteeing [Fuente N] ↔ the
        # FUENTES list 1:1. `orig_to_display` maps the model's index → display N.
        citations: list[dict] = []
        authority_used: set[str] = set()
        orig_to_display: dict[int, int] = {}
        chunkid_to_display: dict[int, int] = {}
        for n in cited_indices:
            try:
                i = int(n)
            except (TypeError, ValueError):
                continue
            if not (1 <= i <= len(chunks)):
                continue
            c = chunks[i - 1]
            if c.chunk_id in chunkid_to_display:
                # A second model-index pointing at an already-cited chunk: reuse
                # its display number so the marker still resolves 1:1.
                orig_to_display[i] = chunkid_to_display[c.chunk_id]
                continue
            display = len(citations) + 1
            citations.append(
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "page_from": c.page_from,
                    "page_to": c.page_to,
                    "authority_level": c.authority_level,
                }
            )
            orig_to_display[i] = display
            chunkid_to_display[c.chunk_id] = display
            if c.authority_level:
                authority_used.add(c.authority_level)

        answer = _renumber_markers(answer, orig_to_display)

        top_score = max((c.score for c in chunks), default=0.0)
        grounded = len(citations) >= 1 and bool(answer)

        # Order authority_used by precedence (convenio first, then baseline) for a
        # stable, readable audit value.
        authority_ordered = sorted(authority_used, key=lambda a: _AUTHORITY_RANK.get(a, 99))

        return SynthesisResult(
            answer=answer,
            citations=citations,
            grounding_signal={
                "grounded": grounded,
                "citation_count": len(citations),
                "top_chunk_score": round(float(top_score), 6),
            },
            confidence=confidence,
            authority_used=authority_ordered,
            trace_fragment={
                "provider": config.provider,
                "model": config.model,
                "prompt_tokens": getattr(resp.usage, "input_tokens", None),
                "completion_tokens": getattr(resp.usage, "output_tokens", None),
                "synthesis_ms": elapsed_ms,
                "authority_used": authority_ordered,
            },
        )

    def classify(
        self,
        question: str,
        api_key: str,
        config: ProviderConfig,
    ) -> RouterResult:
        """Router classification (ADR-0016) with the SMALL/FAST model. Returns a
        label + confidence and, for a compound question, the decomposed
        subqueries. On any parse/transport failure the caller (hr-backend) is
        fail-safe — this method never raises a routing decision it can't justify;
        it returns a low-confidence prose result so hr-backend defaults safely."""
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, base_url=config.endpoint or None)
        started = time.monotonic()
        resp = client.messages.create(
            model=config.model,
            max_tokens=512,
            system=ROUTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Pregunta: {question}"}],
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        raw_text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

        try:
            envelope = _extract_json(raw_text)
        except (json.JSONDecodeError, ValueError):
            # Unparseable → low-confidence prose; hr-backend's fail-safe takes the
            # safe prose+floor path (never a silent misroute).
            return RouterResult(
                label="prose",
                confidence=0.0,
                subqueries=[],
                reason="router_parse_error",
                trace_fragment={"provider": config.provider, "model": config.model, "router_ms": elapsed_ms, "parse_error": True},
            )

        label = str(envelope.get("label", "prose")).strip().lower()
        if label not in ("salary", "prose", "off_domain"):
            label = "prose"
        confidence = float(envelope.get("confidence", 0.0) or 0.0)
        subqueries = [str(s).strip() for s in (envelope.get("subqueries") or []) if str(s).strip()]

        return RouterResult(
            label=label,
            confidence=confidence,
            subqueries=subqueries,
            reason="llm",
            trace_fragment={
                "provider": config.provider,
                "model": config.model,
                "router_ms": elapsed_ms,
                "prompt_tokens": getattr(resp.usage, "input_tokens", None),
                "completion_tokens": getattr(resp.usage, "output_tokens", None),
            },
        )

    def ground(
        self,
        question: str,
        answer: str,
        chunks: list[GroundChunk],
        api_key: str,
        config: ProviderConfig,
    ) -> GroundingResult:
        """Per-claim entailment check (Sprint 2b-2 §5) with the CAPABLE answer
        model (entailment is subtle — never the cheap router model). Table-aware.
        On a parse failure it returns grounded=False (conservative — hr-backend
        escalates rather than surfacing an unverified answer)."""
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, base_url=config.endpoint or None)
        user_prompt = _build_ground_prompt(question, answer, chunks)

        # Call once at the (generous) budget. If the model still stops at the token
        # cap (stop_reason == "max_tokens") the JSON is truncated — retry ONCE at a
        # larger budget before giving up (Correction-04). A truncation is a budget
        # problem, never evidence of a fabricated claim.
        started = time.monotonic()
        resp = client.messages.create(
            model=config.model,
            max_tokens=GROUND_MAX_TOKENS,
            system=GROUND_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        budget = GROUND_MAX_TOKENS
        retried_on_truncation = False
        if getattr(resp, "stop_reason", None) == "max_tokens":
            retried_on_truncation = True
            budget = GROUND_MAX_TOKENS_RETRY
            resp = client.messages.create(
                model=config.model,
                max_tokens=GROUND_MAX_TOKENS_RETRY,
                system=GROUND_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        elapsed_ms = int((time.monotonic() - started) * 1000)

        # Still truncated after the retry → a DISTINCT outcome, not a fabricated
        # claim. hr-backend still escalates (the conservative floor is unchanged),
        # but the trace says grounding_truncated so a truncation is never read as a
        # genuine ungrounded claim.
        if getattr(resp, "stop_reason", None) == "max_tokens":
            return GroundingResult(
                grounded=False,
                claims=[],
                ungrounded=["<grounding check truncated>"],
                trace_fragment={
                    "provider": config.provider,
                    "model": config.model,
                    "ground_ms": elapsed_ms,
                    "grounding_truncated": True,
                    "retried_on_truncation": retried_on_truncation,
                    "max_tokens": budget,
                },
            )

        raw_text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

        try:
            envelope = _extract_json(raw_text)
        except (json.JSONDecodeError, ValueError):
            return GroundingResult(
                grounded=False,
                claims=[],
                ungrounded=["<grounding check unparseable>"],
                trace_fragment={"provider": config.provider, "model": config.model, "ground_ms": elapsed_ms, "parse_error": True, "retried_on_truncation": retried_on_truncation},
            )

        claims_in = envelope.get("claims") or []
        claims: list[dict] = []
        ungrounded: list[str] = []
        substantive_count = 0
        provenance_count = 0
        for c in claims_in:
            if not isinstance(c, dict):
                continue
            text = str(c.get("claim", "")).strip()
            is_grounded = bool(c.get("grounded", False))
            support = c.get("supporting_source")
            # Substantive vs provenance (Correction-01): only PURE provenance
            # statements are exempt from entailment — the citation carries the
            # origin. PRECISION GUARD: anything not explicitly tagged provenance is
            # treated as substantive, so a fabricated rule/figure can never escape
            # the gate by being mislabelled. A claim is exempt ONLY when the model
            # tagged it provenance.
            kind = "provenance" if str(c.get("tipo", "")).strip().lower() in ("procedencia", "provenance") else "substantive"
            claims.append({"claim": text, "kind": kind, "grounded": is_grounded, "supporting_source": support})
            if kind == "provenance":
                provenance_count += 1
                continue
            substantive_count += 1
            if not is_grounded and text:
                ungrounded.append(text)

        # The gate (Correction-01): grounded iff there is at least one SUBSTANTIVE
        # claim and EVERY substantive claim is entailed. Provenance claims are not
        # gated (the [Fuente N] marker + authority badge carry the origin). With
        # zero substantive claims we cannot assert the answer is supported → NOT
        # grounded (conservative direction).
        grounded = substantive_count >= 1 and len(ungrounded) == 0

        return GroundingResult(
            grounded=grounded,
            claims=claims,
            ungrounded=ungrounded,
            trace_fragment={
                "provider": config.provider,
                "model": config.model,
                "ground_ms": elapsed_ms,
                "prompt_tokens": getattr(resp.usage, "input_tokens", None),
                "completion_tokens": getattr(resp.usage, "output_tokens", None),
                "claim_count": len(claims),
                "substantive_count": substantive_count,
                "provenance_count": provenance_count,
                "max_tokens": budget,
                "retried_on_truncation": retried_on_truncation,
            },
        )

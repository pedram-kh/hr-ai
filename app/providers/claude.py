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

from .base import AnswerProvider, ChunkInput, ProviderConfig, SynthesisResult

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
    "2. Cada cifra y cada afirmación debe citarse con [Fuente N] apuntando a la fuente "
    "que REALMENTE enuncia ese dato — nunca a la fuente más cercana o más parecida. Si "
    "una fuente habla de un tema distinto (p. ej. la duración de un contrato formativo) "
    "NO sirve para responder sobre otro tema (p. ej. el periodo de prueba), aunque "
    "ambos mencionen 'meses'.\n"
    "3. REGLA DE PRECEDENCIA: el CONVENIO de la persona trabajadora gobierna los temas "
    "que regula. La LEY NACIONAL (el Estatuto de los Trabajadores) es solo la base "
    "mínima que aplica donde el convenio guarda silencio. Cuando el convenio se "
    "pronuncia sobre el tema preguntado, su respuesta PREVALECE; usa el Estatuto solo "
    "para rellenar lo que el convenio no regula. NUNCA afirmes una regla del convenio "
    "que las fuentes del convenio no contienen.\n"
    "4. Conflicto convenio vs Estatuto sobre la MISMA pregunta: presenta la del "
    "convenio como la que gobierna y menciona la del Estatuto solo como base general; "
    "si no puedes determinar con seguridad cuál gobierna, indícalo y pon confianza "
    "baja. NUNCA mezcles ambas en una sola cifra ni presentes la base nacional como si "
    "fuera la respuesta cuando el convenio regula el tema.\n"
    "5. CUANDO LAS FUENTES NO RESPONDEN la pregunta: \n"
    "   - Si una fuente de LEY NACIONAL (Estatuto) sí la responde, responde a partir de "
    "ella (citándola).\n"
    "   - Si NINGUNA fuente la responde, NO inventes y NO respondas: devuelve "
    "`\"cited_sources\": []` y `\"confidence\"` baja (≤ 0.2), con un `answer` breve que "
    "diga que no dispones de información suficiente. Es PREFERIBLE abstenerse a dar una "
    "respuesta no fundamentada — la abstención hace que el caso se derive a una persona.\n"
    "6. Responde en el MISMO idioma que la pregunta.\n\n"
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
        "para lo que el convenio no cubra. Responde en el idioma de la pregunta y "
        "devuelve solo el JSON."
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
        citations: list[dict] = []
        authority_used: set[str] = set()
        seen_chunk_ids: set[int] = set()
        for n in cited_indices:
            try:
                i = int(n)
            except (TypeError, ValueError):
                continue
            if 1 <= i <= len(chunks):
                c = chunks[i - 1]
                if c.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(c.chunk_id)
                citations.append(
                    {
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "page_from": c.page_from,
                        "page_to": c.page_to,
                        "authority_level": c.authority_level,
                    }
                )
                if c.authority_level:
                    authority_used.add(c.authority_level)

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

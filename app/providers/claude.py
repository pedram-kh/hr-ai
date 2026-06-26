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
    ConvenioCandidate,
    GroundChunk,
    GroundingResult,
    ProviderConfig,
    RouterResult,
    SegmentedFactsResult,
    SynthesisResult,
    TagProposalResult,
    VocabularyCandidate,
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


def _salvage_facts(text: str) -> dict:
    """Recover complete fact objects from a `{"facts":[...]}` blob the model
    truncated mid-array (stop_reason="max_tokens"). Scans the array brace-by-brace
    (string-aware) and keeps only fully-closed objects; the trailing partial object
    is discarded. Closed-set validation still applies downstream, so a salvaged
    partial is strictly safer than the alternative (a parse failure → zero facts)."""
    start = text.find('"facts"')
    if start == -1:
        return {"facts": []}
    lb = text.find("[", start)
    if lb == -1:
        return {"facts": []}

    facts: list = []
    depth = 0
    obj_start: int | None = None
    in_str = False
    esc = False
    for i in range(lb + 1, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    facts.append(json.loads(text[obj_start : i + 1]))
                except (json.JSONDecodeError, ValueError):
                    pass
                obj_start = None
        elif ch == "]" and depth == 0:
            break
    return {"facts": facts}


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

# Output-token budget for the segmentation JSON (Sprint 7b-2). A multi-province
# periodo file emits one verbose object per scope (value + raw_values + full
# source_excerpt); a ~45-fact file (e.g. PERÍODOS 2026) needs ~16-18k tokens. The
# original 8192 cap truncated that file mid-array (stop_reason="max_tokens"),
# `_extract_json` failed, and the source produced ZERO facts. 32000 gives ample
# headroom (well within Sonnet's output limit); `_salvage_facts` is the residual
# safety net so a truncation degrades to "most facts" rather than "no facts".
SEGMENT_MAX_TOKENS = 32000


# --- Tagging tier (Sprint 7a, ADR-0011/0020) — read content, PROPOSE facets ----
# The AI is a STRICT, INERT proposer: it suggests document-level facets bound to
# the CLOSED vocabulary hr-backend passes, and flags anything it cannot resolve
# as a raw_unmatched_value (with an optional variant hint) — it NEVER invents a
# vocabulary value, and it does DOCUMENT-LEVEL facet tagging only (never splits a
# multi-scope file into per-scope facts — that is Sprint 7b).
TAG_PROPOSAL_SYSTEM_PROMPT = (
    "Eres un asistente de catalogación documental para una plataforma de RR. HH. "
    "que gestiona convenios colectivos españoles. Recibes el TEXTO de UN documento "
    "(que el parser de nombre de archivo no pudo clasificar) y unas LISTAS de "
    "vocabulario CONTROLADO (convenios, territorios, sectores, tipos de documento). "
    "Tu tarea: PROPONER las facetas de ESTE documento, cada una con su confianza.\n\n"
    "REGLAS ABSOLUTAS:\n"
    "1. SOLO PROPONES. No decides nada; un humano revisa y verifica tu propuesta. "
    "Puedes equivocarte sin causar daño — tu salida es inerte hasta que un humano la "
    "verifique.\n"
    "2. VOCABULARIO CERRADO: para convenio, territorio, sector y tipo de documento, "
    "devuelve EXCLUSIVAMENTE el `id` (o `code` para tipo de documento) de un valor de "
    "las listas proporcionadas. JAMÁS inventes un valor nuevo ni devuelvas texto "
    "libre. Si el documento parece referirse a un valor que NO está en las listas, "
    "NO lo inventes: regístralo en `raw_unmatched_values` con el texto literal y, si "
    "se parece a un valor existente, indícalo en `variant_of` (el id y por qué).\n"
    "3. ÁMBITO POR DOCUMENTO: clasificas el documento COMPLETO con UN conjunto de "
    "facetas. NO segmentes el documento en hechos por ámbito ni produzcas varias "
    "filas de ámbito (eso es otra fase). Un documento → un conjunto de facetas.\n"
    "4. El territorio y el sector se derivan del convenio: si propones un convenio, "
    "propón su territorio/sector coherentes; si no hay convenio claro, puedes dejar "
    "esas facetas sin proponer (confianza baja) en lugar de adivinar.\n"
    "5. validity: propón el rango de vigencia como cadena \"AAAA-MM-DD..AAAA-MM-DD\" "
    "solo si el texto lo enuncia con claridad; si no, omítela.\n"
    "6. topics: propón SOLO ids de la lista de topics APROBADOS proporcionada (si la "
    "hay); nunca inventes un topic.\n"
    "7. Sé honesto con la confianza (0..1): baja cuando el texto es ambiguo o un escaneo "
    "pobre. La confianza global es el MÍNIMO de las facetas.\n\n"
    "FORMATO DE SALIDA: devuelve EXCLUSIVAMENTE un objeto JSON válido, sin texto "
    "alrededor, con esta forma:\n"
    '{"facets": [{"facet": "document_type", "value_code": "<code>", "confidence": <0..1>}, '
    '{"facet": "convenio", "value_id": <id>, "confidence": <0..1>}, '
    '{"facet": "territory", "value_id": <id>, "confidence": <0..1>}, '
    '{"facet": "sector", "value_id": <id>, "confidence": <0..1>}, '
    '{"facet": "validity", "value": "AAAA-MM-DD..AAAA-MM-DD", "confidence": <0..1>}], '
    '"topics": [{"topic_id": <id>, "confidence": <0..1>}], '
    '"raw_unmatched_values": [{"facet": "sector", "value": "<texto literal>", '
    '"variant_of": {"id": <id>, "reason": "<por qué se parece>"}}]}'
)

# Cap the document text handed to the model so a 100-page scan can't blow the
# prompt budget. The opening pages carry the title/scope signals the parser
# needs; this is a proposal, not retrieval, so the head is sufficient.
TAG_PROPOSAL_TEXT_CAP = 12000


def _facet_candidate_block(label: str, items: list[VocabularyCandidate], with_code: bool = False) -> str:
    lines = [f"{label}:"]
    for c in items:
        alias = f" (alias: {', '.join(c.aliases)})" if c.aliases else ""
        ident = f"code={c.code}" if with_code and c.code is not None else f"id={c.id}"
        lines.append(f"  - {ident} · {c.name}{alias}")
    return "\n".join(lines)


def _build_tag_prompt(page_text: str, candidate_vocabulary: dict[str, list[VocabularyCandidate]]) -> str:
    text = (page_text or "").strip()
    if len(text) > TAG_PROPOSAL_TEXT_CAP:
        text = text[:TAG_PROPOSAL_TEXT_CAP] + "\n…[texto truncado]"
    blocks = ["TEXTO DEL DOCUMENTO:", text, "", "VOCABULARIO CONTROLADO (usa solo estos ids/códigos):"]
    if candidate_vocabulary.get("document_types"):
        blocks.append(_facet_candidate_block("Tipos de documento", candidate_vocabulary["document_types"], with_code=True))
    if candidate_vocabulary.get("territories"):
        blocks.append(_facet_candidate_block("Territorios", candidate_vocabulary["territories"]))
    if candidate_vocabulary.get("sectors"):
        blocks.append(_facet_candidate_block("Sectores", candidate_vocabulary["sectors"]))
    if candidate_vocabulary.get("convenios"):
        blocks.append(_facet_candidate_block("Convenios (preseleccionados por indicios)", candidate_vocabulary["convenios"]))
    if candidate_vocabulary.get("topics"):
        blocks.append(_facet_candidate_block("Topics aprobados", candidate_vocabulary["topics"]))
    blocks.append("")
    blocks.append(
        "Propón las facetas del documento vinculándolas SOLO a estos ids/códigos. "
        "Lo que no resuelvas, regístralo en raw_unmatched_values (nunca inventes). "
        "Devuelve solo el JSON."
    )
    return "\n".join(blocks)


# --- Segmentation agent (Sprint 7b-2, ADR-0022) — read a multi-scope reference
# source, SEGMENT into per-scope facts, BIND each to a real convenio. The single
# riskiest cognition in the project: a wrong scope is a confident, exact, wrong
# answer. The load-bearing instruction is HEADER-CARRY (scope resets on each
# TERRITORY/SECTOR header). The AI is a strict, inert proposer (ai_agent/
# needs_review; never answerable, never salary, never new vocabulary).
SEGMENT_FACTS_SYSTEM_PROMPT = (
    "Eres un agente de segmentación documental para una plataforma de RR. HH. que "
    "gestiona convenios colectivos españoles. Recibes el TEXTO COMPLETO de UN "
    "documento de referencia (p. ej. una recopilación de periodos de prueba por "
    "provincia y sector) y una lista de CONVENIOS de vocabulario CONTROLADO (cada "
    "uno con su territorio y sector derivados, y sus categorías profesionales si "
    "las hay). Tu tarea: PARTIR el documento en HECHOS individuales, uno por "
    "ÁMBITO, y asignar a cada hecho su convenio.\n\n"
    "REGLA MÁS IMPORTANTE — ARRASTRE DE ENCABEZADOS (header-carry):\n"
    "El documento se organiza como ENCABEZADO DE TERRITORIO (una provincia o "
    "ámbito: ESTATAL, ÁLAVA, NAVARRA, GIPUZKOA…), luego ENCABEZADO DE SECTOR (un "
    "sector o un convenio nombrado: COEAS, Hostelería, Intervención Social, "
    "Oficinas y Despachos…), y debajo las LÍNEAS DE VALOR (p. ej. 'Grupo 1: Cinco "
    "meses'). El ámbito de CADA línea de valor es el TERRITORIO + SECTOR vigentes "
    "MÁS RECIENTES. REINICIA el SECTOR en cada nuevo encabezado de sector, y "
    "REINICIA territorio Y sector en cada nuevo encabezado de territorio. NUNCA "
    "dejes que una línea herede la provincia del bloque anterior. Los encabezados "
    "NO siempre tienen un estilo distinto: reconócelos por el CONTENIDO (un nombre "
    "de provincia/ámbito = territorio; un nombre de sector o una línea corta que "
    "introduce un bloque de 'Grupo …' = sector). DEDUCE la jerarquía del TEXTO.\n\n"
    "REGLAS ABSOLUTAS:\n"
    "1. SOLO PROPONES. No decides nada; un humano revisa y verifica. Tu salida es "
    "INERTE hasta que un humano la verifique — puedes equivocarte sin causar daño, "
    "salvo que des un ámbito equivocado con alta confianza (ese es el peor error). "
    "Ante la duda del ámbito, BAJA la confianza y rellena `uncertainty`.\n"
    "2. VOCABULARIO CERRADO: vincula cada hecho a un `convenio_id` de la lista. "
    "JAMÁS inventes un convenio ni devuelvas texto libre como id. COEAS equivale a "
    "'Ocio Educativo y Animación Sociocultural'. Resuelve variantes ortográficas "
    "(Gipuzkoa/Guipúzcoa, Bizkaia/Vizcaya) por los alias. Si el (territorio, "
    "sector) de un bloque NO coincide con ningún convenio de la lista (p. ej. "
    "'ámbito estatal cuando no hay convenio territorial' o una regla general del "
    "Estatuto de los Trabajadores), NO fuerces un convenio: omite el hecho o "
    "emítelo con `uncertainty.field='scope'`. Mejor marcar incierto que adivinar.\n"
    "3. UN HECHO POR ÁMBITO (valores múltiples): un bloque 'Grupo X' es UN hecho, "
    "aunque su desglose por tipo de contrato ocupe varias líneas (p. ej. "
    "'Indefinido: 90 días / Temporal +3m: 75 / Temporal −3m: 60'). Mete el "
    "desglose COMPLETO en `value` y en `raw_values` — NO crees tres hechos. Un "
    "rango de grupos ('Grupos 1 y 2', 'Grupos 3,4,5 y 6') es UN hecho para ese "
    "rango. El ÁMBITO es la unidad consultable; el desglose vive dentro.\n"
    "4. group_label OBLIGATORIO: devuelve SIEMPRE el grupo TAL CUAL aparece "
    "('Grupo 1', 'Grupo 2 (resto áreas)', 'Grupo 1 y área cinco de Grupo 2', "
    "'Obreros y subalternos'). Es el discriminador de identidad del hecho. Si una "
    "categoría de la lista del convenio coincide claramente con el grupo, pon "
    "también `job_category_id`; si no, déjalo null (lo normal) — el grupo va en "
    "group_label.\n"
    "5. EXPRESIONES DE GRUPO COMPUESTAS ('Grupo 1 y área cinco de Grupo 2') no "
    "mapean a una sola categoría: deja `job_category_id` null y marca "
    "`uncertainty.field='group'` con el motivo.\n"
    "6. topic: vincula `topic_id` al topic 'periodo de prueba' de la lista si está "
    "presente; si no, déjalo null. Nunca inventes un topic.\n"
    "7. NADA DE SALARIOS: ignora por completo tablas de salarios, €/hora, SMI y "
    "rejillas de retribución (van por otra vía). En una hoja de cálculo salarial, "
    "los ÚNICOS hechos de referencia admisibles son jornada (horas/año, "
    "horas/semana) o vigencia, y SOLO si puedes asignarlos con certeza a un "
    "convenio. Si todo es salario, devuelve `facts` vacío — es lo correcto.\n"
    "8. NO propongas vigencia ni fechas (las fija el sistema). NO propongas "
    "autoridad.\n"
    "9. TRAZABILIDAD OBLIGATORIA: cada hecho lleva `source_excerpt` = la(s) "
    "línea(s) EXACTAS de origen con su rastro de encabezados (p. ej. 'ALAVA › "
    "COEAS ALAVA › Grupo 1: Cinco meses') para que el revisor compruebe el ámbito "
    "contra la cita. Usa los marcadores [loc:…] del texto para `source_locator`.\n"
    "10. confianza (0..1): honesta. Baja cuando el ámbito es ambiguo. "
    "`uncertainty` = {field, reason} cuando dudes (field ∈ scope|group|version|"
    "value); null si estás seguro.\n"
    "11. ÁMBITO ESTATAL SUPLETORIO / REGLA GENERAL → NO VINCULES, marca incierto. "
    "Cuando un bloque se presenta como ámbito ESTATAL supletorio o como regla "
    "general que aplica «cuando NO hay convenio territorial específico» (p. ej. "
    "«ámbito estatal (cuando no hay convenio territorial específico)», «Estatuto de "
    "los Trabajadores», «consideración general», o cualquier redacción que diga que "
    "se aplica a falta de un acuerdo territorial), NO lo vincules a ningún "
    "convenio, AUNQUE exista en la lista un convenio estatal real y del nivel "
    "correcto. Emítelo con `uncertainty.field='scope'` y reason='statutory fallback "
    "— no territorial convenio applies', dejando el ámbito SIN vincular, u "
    "OMÍTELO. Vincular con confianza aquí es un error aunque el convenio elegido "
    "sea real y del nivel correcto: es justo el caso que debe quedar para el "
    "juicio humano.\n"
    "12. CONTENIDO NO-PERIODO EN ESTA VÍA → NO EMITAS NADA. Si la fuente es una "
    "hoja de salarios/jornada/horas (sus líneas son retribuciones, horas, días de "
    "vacaciones, SMI… y NO reglas de periodo de prueba), NO emitas ningún hecho "
    "salvo que una línea enuncie CLARAMENTE una regla de PERIODO DE PRUEBA. Las "
    "cifras de jornada/horas/vacaciones NO son hechos de referencia en esta vía: "
    "no vincules nada para ellas. (Es la contención de la regla 7: la invariante ya "
    "bloquea las filas salariales; esto impide además disfrazar de hecho una línea "
    "de jornada.)\n\n"
    "FORMATO DE SALIDA: devuelve EXCLUSIVAMENTE un objeto JSON válido, sin texto "
    "alrededor, con esta forma:\n"
    '{"facts": [{"convenio_id": <id>, "job_category_id": <id|null>, '
    '"group_label": "<grupo tal cual|null>", "topic_id": <id|null>, '
    '"value": "<regla legible, con el desglose dentro>", '
    '"raw_values": {<estructura literal opcional>}, "confidence": <0..1>, '
    '"uncertainty": {"field": "scope|group|version|value", "reason": "<por qué>"}, '
    '"source_locator": "<loc>", "source_excerpt": "<línea(s) exactas + rastro>"}]}'
)

# The reference fixtures are tiny (78/95 paragraphs); feed the FULL concatenated
# text so header-carry sees the whole sequence (Q9 — never chunk). The cap only
# guards against a huge salary xlsx (the routing test wants ~zero facts anyway).
SEGMENT_TEXT_CAP = 48000


def _convenio_candidate_block(convenios: list[ConvenioCandidate]) -> str:
    lines = ["Convenios (vincula cada hecho a uno de estos id — territorio/sector se derivan):"]
    for c in convenios:
        alias = f" (alias: {', '.join(c.aliases)})" if c.aliases else ""
        terr_alias = f" [{', '.join(c.territory_aliases)}]" if c.territory_aliases else ""
        lines.append(
            f"  - id={c.id} · {c.name}{alias} · territorio: {c.territory_name}{terr_alias}"
            f" · sector: {c.sector_name}"
        )
        for jc in c.job_categories:
            gc = f" ({jc.group_code})" if jc.group_code else ""
            lines.append(f"      · categoría id={jc.id}: {jc.name}{gc}")
    return "\n".join(lines)


def _build_segment_prompt(
    pages_text: str,
    candidate_convenios: list[ConvenioCandidate],
    candidate_topics: list[VocabularyCandidate],
) -> str:
    text = (pages_text or "").strip()
    if len(text) > SEGMENT_TEXT_CAP:
        text = text[:SEGMENT_TEXT_CAP] + "\n…[texto truncado]"
    blocks = [
        "TEXTO COMPLETO DEL DOCUMENTO DE REFERENCIA "
        "(respeta el arrastre de encabezados territorio→sector→grupo):",
        text,
        "",
        "VOCABULARIO CONTROLADO (usa solo estos ids):",
        _convenio_candidate_block(candidate_convenios),
    ]
    if candidate_topics:
        blocks.append(_facet_candidate_block("Topics aprobados", candidate_topics))
    blocks.append("")
    blocks.append(
        "Segmenta el documento en hechos por ámbito, uno por (convenio + grupo), "
        "vinculando cada uno a un convenio_id de la lista. Arrastra y REINICIA el "
        "ámbito en cada encabezado. No inventes convenios; marca incierto lo que "
        "no resuelvas. Ignora el salario. Devuelve solo el JSON."
    )
    return "\n".join(blocks)


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

    def propose_tags(
        self,
        page_text: str,
        candidate_vocabulary: dict[str, list[VocabularyCandidate]],
        api_key: str,
        config: ProviderConfig,
    ) -> TagProposalResult:
        """Read the document text and PROPOSE document-level facets bound to the
        provided closed vocabulary (Sprint 7a). Strict, inert proposer — it
        returns suggestions only; hr-backend persists them as `ai_agent`
        provenance and keeps the doc `under_review` (the embedding gate). On a
        parse failure it returns an empty proposal at confidence 0 so the doc
        simply stays in the human queue (never a silent bad tag)."""
        import anthropic  # lazy — dep only needed at call time

        client = anthropic.Anthropic(api_key=api_key, base_url=config.endpoint or None)
        user_prompt = _build_tag_prompt(page_text, candidate_vocabulary)

        started = time.monotonic()
        resp = client.messages.create(
            model=config.model,
            max_tokens=1024,
            system=TAG_PROPOSAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        raw_text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

        try:
            envelope = _extract_json(raw_text)
        except (json.JSONDecodeError, ValueError):
            return TagProposalResult(
                facets=[],
                topics=[],
                raw_unmatched_values=[],
                overall_confidence=0.0,
                trace_fragment={
                    "provider": config.provider,
                    "model": config.model,
                    "propose_ms": elapsed_ms,
                    "parse_error": True,
                },
            )

        # Validate the model's facet references against the provided candidate ids /
        # codes so a hallucinated id can NEVER reach hr-backend. An out-of-set
        # reference is dropped here (the closed-vocabulary guarantee, ADR-0002).
        valid_ids: dict[str, set[int]] = {
            "convenio": {c.id for c in candidate_vocabulary.get("convenios", [])},
            "territory": {c.id for c in candidate_vocabulary.get("territories", [])},
            "sector": {c.id for c in candidate_vocabulary.get("sectors", [])},
        }
        valid_doctype_codes = {c.code for c in candidate_vocabulary.get("document_types", []) if c.code}
        valid_topic_ids = {c.id for c in candidate_vocabulary.get("topics", [])}

        facets: list[dict] = []
        confidences: list[float] = []
        for f in envelope.get("facets") or []:
            if not isinstance(f, dict):
                continue
            facet = str(f.get("facet", "")).strip()
            try:
                conf = float(f.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            if facet == "document_type":
                code = f.get("value_code")
                if code in valid_doctype_codes:
                    facets.append({"facet": facet, "value_code": code, "confidence": conf})
                    confidences.append(conf)
            elif facet in ("convenio", "territory", "sector"):
                vid = f.get("value_id")
                if isinstance(vid, int) and vid in valid_ids[facet]:
                    facets.append({"facet": facet, "value_id": vid, "confidence": conf})
                    confidences.append(conf)
            elif facet == "validity":
                val = str(f.get("value", "")).strip()
                if val:
                    facets.append({"facet": facet, "value": val, "confidence": conf})
                    confidences.append(conf)

        topics: list[dict] = []
        for t in envelope.get("topics") or []:
            if not isinstance(t, dict):
                continue
            tid = t.get("topic_id")
            if isinstance(tid, int) and tid in valid_topic_ids:
                try:
                    tconf = float(t.get("confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    tconf = 0.0
                topics.append({"topic_id": tid, "confidence": tconf})

        raw_unmatched: list[dict] = []
        for r in envelope.get("raw_unmatched_values") or []:
            if not isinstance(r, dict):
                continue
            facet = str(r.get("facet", "")).strip()
            value = str(r.get("value", "")).strip()
            if not facet or not value:
                continue
            entry: dict = {"facet": facet, "value": value}
            variant = r.get("variant_of")
            # Keep a variant hint ONLY when it points at a real existing id.
            if isinstance(variant, dict) and isinstance(variant.get("id"), int):
                vfacet_ids = valid_ids.get(facet, set())
                if variant["id"] in vfacet_ids:
                    entry["variant_of"] = {"id": variant["id"], "reason": str(variant.get("reason", "")).strip()}
            raw_unmatched.append(entry)

        overall = round(min(confidences), 3) if confidences else 0.0

        return TagProposalResult(
            facets=facets,
            topics=topics,
            raw_unmatched_values=raw_unmatched,
            overall_confidence=overall,
            trace_fragment={
                "provider": config.provider,
                "model": config.model,
                "propose_ms": elapsed_ms,
                "prompt_tokens": getattr(resp.usage, "input_tokens", None),
                "completion_tokens": getattr(resp.usage, "output_tokens", None),
                "facet_count": len(facets),
            },
        )

    def segment_facts(
        self,
        pages_text: str,
        candidate_convenios: list[ConvenioCandidate],
        candidate_topics: list[VocabularyCandidate],
        api_key: str,
        config: ProviderConfig,
    ) -> SegmentedFactsResult:
        """Segment a multi-scope reference source into per-scope facts bound to
        the closed convenio vocabulary (Sprint 7b-2). Strict, inert proposer. On
        a parse failure it returns an empty facts list so the source simply stays
        unsegmented in the human queue (never a silent bad fact)."""
        import anthropic  # lazy — dep only needed at call time

        client = anthropic.Anthropic(api_key=api_key, base_url=config.endpoint or None)
        user_prompt = _build_segment_prompt(pages_text, candidate_convenios, candidate_topics)

        started = time.monotonic()
        # A multi-province periodo file yields ~30-50 verbose facts of JSON; the
        # large SEGMENT_MAX_TOKENS budget needed to avoid a mid-array truncation
        # exceeds the SDK's non-streaming ceiling ("Streaming is required for
        # operations that may take longer than 10 minutes"), so this call streams.
        raw_text = ""
        with client.messages.stream(
            model=config.model,
            max_tokens=SEGMENT_MAX_TOKENS,
            system=SEGMENT_FACTS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for chunk in stream.text_stream:
                raw_text += chunk
            resp = stream.get_final_message()
        elapsed_ms = int((time.monotonic() - started) * 1000)
        truncated = getattr(resp, "stop_reason", None) == "max_tokens"

        salvaged = False
        try:
            envelope = _extract_json(raw_text)
        except (json.JSONDecodeError, ValueError):
            # A truncated array (or stray prose) fails strict parse — recover every
            # complete object rather than dropping the whole source to zero facts.
            envelope = _salvage_facts(raw_text)
            salvaged = True
            if not envelope.get("facts"):
                return SegmentedFactsResult(
                    facts=[],
                    trace_fragment={
                        "provider": config.provider,
                        "model": config.model,
                        "segment_ms": elapsed_ms,
                        "parse_error": True,
                    },
                )

        # Closed-set validation (ADR-0011 by construction): a hallucinated
        # convenio/category/topic id can NEVER reach hr-backend. A category is
        # valid ONLY for its own convenio (mirrors the hr-backend belongs-to
        # check) — otherwise it is dropped to null, not silently mis-bound.
        valid_convenio_ids = {c.id for c in candidate_convenios}
        categories_by_convenio: dict[int, set[int]] = {
            c.id: {jc.id for jc in c.job_categories} for c in candidate_convenios
        }
        valid_topic_ids = {c.id for c in candidate_topics}

        facts: list[dict] = []
        for f in envelope.get("facts") or []:
            if not isinstance(f, dict):
                continue
            cid = f.get("convenio_id")
            if not isinstance(cid, int) or cid not in valid_convenio_ids:
                continue  # never accept an unbound/hallucinated scope
            value = str(f.get("value", "")).strip()
            if not value:
                continue  # a fact must carry a rule

            jcid = f.get("job_category_id")
            if not (isinstance(jcid, int) and jcid in categories_by_convenio.get(cid, set())):
                jcid = None  # drop a category that doesn't belong to this convenio

            tid = f.get("topic_id")
            if not (isinstance(tid, int) and tid in valid_topic_ids):
                tid = None

            try:
                conf = float(f.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            uncertainty = None
            u = f.get("uncertainty")
            if isinstance(u, dict) and (u.get("field") or u.get("reason")):
                uncertainty = {
                    "field": str(u.get("field", "")).strip() or "scope",
                    "reason": str(u.get("reason", "")).strip(),
                }

            group_label = f.get("group_label")
            group_label = str(group_label).strip() if group_label not in (None, "") else None

            raw_values = f.get("raw_values")
            if not isinstance(raw_values, dict):
                raw_values = None

            facts.append(
                {
                    "convenio_id": cid,
                    "job_category_id": jcid,
                    "group_label": group_label,
                    "topic_id": tid,
                    "value": value,
                    "raw_values": raw_values,
                    "confidence": round(conf, 3),
                    "uncertainty": uncertainty,
                    "source_locator": (str(f.get("source_locator", "")).strip() or None),
                    "source_excerpt": str(f.get("source_excerpt", "")).strip(),
                }
            )

        return SegmentedFactsResult(
            facts=facts,
            trace_fragment={
                "provider": config.provider,
                "model": config.model,
                "segment_ms": elapsed_ms,
                "prompt_tokens": getattr(resp.usage, "input_tokens", None),
                "completion_tokens": getattr(resp.usage, "output_tokens", None),
                "fact_count": len(facts),
                "truncated": truncated,
                "salvaged": salvaged,
            },
        )

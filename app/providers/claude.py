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
    ExplainResult,
    GroundChunk,
    GroundingResult,
    GroupProposalResult,
    OcrPageResult,
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
    # A verified structured reference fact (Sprint 7c, ADR-0023) is convenio-derived
    # structured knowledge: it ranks BELOW the convenio prose/ruling that governs
    # the topic, and ABOVE the national_law (Estatuto) baseline. It can never
    # outrank the convenio.
    "structured_reference": 1,
    "national_law": 2,
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
    "7. NO CALCULES CIFRAS. Enuncia únicamente las cantidades que aparecen "
    "LITERALMENTE en una fuente. Está PROHIBIDO dividir, multiplicar, prorratear, "
    "sumar, aplicar un porcentaje o convertir una cifra en otra (p. ej. pasar de un "
    "importe anual a uno mensual, de mensual a por hora, o aplicar una subida "
    "porcentual): una cifra calculada no consta en ninguna fuente y no puede "
    "citarse. Si la persona pregunta por una cantidad que las fuentes no enuncian "
    "tal cual, di qué cantidad SÍ consta y omite la calculada (ADR-0027: una cifra "
    "es una celda de origen o no se da).\n"
    "8. Responde en el MISMO idioma que la pregunta.\n\n"
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
        # Sprint 7c (ADR-0023): a verified structured datum derived from the
        # convenio — gobierna por debajo del texto del convenio, por encima del
        # Estatuto. The convenio prose still governs on a same-point conflict.
        "structured_reference": "dato de referencia estructurado del convenio (verificado)",
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


# --- Group structure tier (Sprint 7f, ADR-0028) — read ONE convenio, PROPOSE ---
# its group tree. This is the vocabulary the answer path will later compare
# EXACTLY (`convenio_groups.id`, an integer), replacing the bare-digit regex that
# cannot express "área 5 of Grupo 2". No AI runs at answer time; this runs once,
# per convenio, and a human approves every node before anything is comparable.
PROPOSE_GROUPS_SYSTEM_PROMPT = (
    "Eres un analista de convenios colectivos españoles. Tu tarea es leer el texto de UN convenio "
    "y proponer su ESTRUCTURA DE GRUPOS PROFESIONALES: los grupos y, cuando corresponda, las áreas "
    "dentro de un grupo.\n"
    "\n"
    "NO DECIDES NADA. Todo lo que devuelvas es una PROPUESTA que una persona revisará y aprobará "
    "o rechazará. Nada de lo que digas se aplica automáticamente. Por eso es mejor proponer algo "
    "marcado como incierto que callarlo, y siempre es mejor no proponer que inventar.\n"
    "\n"
    "═══ LA REGLA MÁS IMPORTANTE: LA GRANULARIDAD SIGUE A LOS VALORES ═══\n"
    "Propón un ÁREA dentro de un grupo SOLO SI el texto asigna VALORES DISTINTOS a esas partes "
    "del grupo. Ejemplo real: si el convenio dice que el periodo de prueba del Grupo 2 es de 90 "
    "días para el área 5 y de 60 días para el resto de áreas, entonces el Grupo 2 tiene dos áreas "
    "('área 5' y 'resto áreas') porque el convenio las trata de forma distinta.\n"
    "Si el convenio solo MENCIONA áreas, o las enumera sin darles condiciones distintas, propón "
    "UN SOLO nodo para el grupo y NINGUNA área. Dividir un grupo que el convenio no divide es un "
    "error grave: obliga al sistema a exigir una distinción que el convenio nunca hizo, y una "
    "pregunta que hoy se responde bien pasaría a derivarse a una persona.\n"
    "Solo hay DOS niveles: grupo y área dentro de un grupo. Nunca un área dentro de un área.\n"
    "\n"
    "═══ CADA NODO NECESITA UNA CITA ═══\n"
    "Todo nodo lleva `source_excerpt`: la línea o líneas EXACTAS del convenio que lo justifican, "
    "copiadas literalmente, y `source_locator` (p. ej. 'p.12'). Un ÁREA SIN CITA NO ES VÁLIDA: si "
    "no puedes citar dónde el convenio da un valor distinto a esa parte del grupo, no propongas "
    "el área. No parafrasees la cita ni la reconstruyas de memoria.\n"
    "\n"
    "═══ LAS ETIQUETAS SE COPIAN, NO SE NORMALIZAN ═══\n"
    "`code_label` es la etiqueta TAL CUAL la imprime el convenio: 'Grupo 2', 'Grupo I', "
    "'Obreros y subalternos', 'área 5', 'resto áreas'. No la traduzcas, no la numeres, no la "
    "conviertas de romano a árabe, no la pongas en minúsculas ni le inventes un código. Otro "
    "componente se encarga de eso. Muchos convenios no numeran sus grupos: "
    "'Técnicos titulados' o 'Personal técnico y administrativo' son grupos perfectamente válidos.\n"
    "`code_label` es el IDENTIFICADOR del grupo, NO su titular completo. Si el convenio escribe "
    "'Grupo I. Personal directivo' o 'Grupo III: personal de atención directa', el `code_label` es "
    "'Grupo I' / 'Grupo III' y la descripción va en `source_excerpt`, donde es útil para quien "
    "revisa. Esto importa: el mismo convenio suele numerar el grupo en su articulado ('grupo 1') y "
    "titularlo en su anexo ('Grupo I. Personal directivo'), y son EL MISMO grupo — un solo nodo. "
    "Solo cuando el grupo NO tiene número ni romano (p. ej. 'Técnicos titulados') el nombre ES el "
    "identificador.\n"
    "Un nodo es un ÁREA cuando `parent_code_label` es la etiqueta de su grupo; es un GRUPO cuando "
    "`parent_code_label` es null.\n"
    "SIEMPRE emite el grupo padre COMO NODO PROPIO, además de sus áreas. Un grupo dividido no "
    "queda representado por sus áreas: sigue siendo un grupo al que un dato o una persona puede "
    "referirse en conjunto, y su `parent_code_label` debe aparecer literalmente como el "
    "`code_label` de un nodo raíz de tu propia lista. Si divides el Grupo 2 en 'área 5' y "
    "'resto áreas', devuelve TRES nodos: 'Grupo 2', 'área 5' (padre 'Grupo 2') y 'resto áreas' "
    "(padre 'Grupo 2').\n"
    "NO propongas un nodo compuesto: si el convenio dice 'Grupos 1 y 2', eso NO es un grupo "
    "llamado 'Grupos 1 y 2' — son dos grupos, 'Grupo 1' y 'Grupo 2'. Que un mismo artículo les "
    "dé el mismo valor se resolverá después vinculando ese dato a los dos grupos.\n"
    "\n"
    "═══ LAS CATEGORÍAS SON UN CONJUNTO CERRADO ═══\n"
    "En `job_category_ids` puedes adjuntar a un nodo categorías profesionales que YA EXISTEN, "
    "usando sus ids de la lista que se te da. NUNCA inventes una categoría, ni propongas crearla, "
    "ni devuelvas un id que no esté en la lista. Si una categoría no encaja en ningún grupo, "
    "déjala fuera: no adjuntarla es una respuesta válida y frecuente.\n"
    "Muchas filas de esa lista NO son categorías reales — son importes salariales, años, o "
    "conceptos de nómina ('Plus transporte', 'Nocturnidad', 'Coordinacion'). Esas NO se adjuntan "
    "a ningún grupo.\n"
    "El campo `group_code` de esas categorías es solo un INDICIO de cómo estaba maquetada una hoja "
    "de cálculo: en el corpus real la mayoría está vacío y varios contienen importes o años. "
    "Úsalo como pista, nunca como verdad, y si el texto del convenio lo contradice, IGNÓRALO. "
    "El texto del convenio es la única fuente.\n"
    "\n"
    "═══ ETIQUETAS YA EN USO ═══\n"
    "Se te dan las etiquetas de grupo que ya usan datos de referencia VERIFICADOS por una persona "
    "para este convenio. Son una LISTA DE COMPROBACIÓN, no una fuente: si tu estructura no puede "
    "expresar una de ellas, es señal de que te falta un grupo o un área, porque ese dato "
    "verificado quedaría sin poder vincularse. Aun así, cada nodo que propongas debe justificarse "
    "con una cita del convenio, no con la etiqueta.\n"
    "\n"
    "═══ CUANDO EL CONVENIO IMPRIME EL GRUPO UNA SOLA VEZ POR BLOQUE ═══\n"
    "Es habitual que una tabla imprima 'Grupo III:' como cabecera y luego varias filas sin repetir "
    "el grupo. Puedes proponer que esas filas pertenecen a ese grupo, PERO: cita el bloque "
    "(la cabecera y las filas que abarca) en `source_excerpt` y marca "
    "`uncertainty` = {\"field\": \"membership\", \"reason\": \"...\"}. Es una hipótesis razonable "
    "sobre la maquetación, no un hecho del texto, y quien revise debe verlo como tal.\n"
    "\n"
    "═══ INCERTIDUMBRE: SEÑÁLALA, NO LA RESUELVAS ═══\n"
    "`uncertainty` es {\"field\": \"structure|membership|area|label\", \"reason\": \"<por qué>\"} "
    "o null. Si dudas de si algo es un grupo o una categoría, si el texto está dañado por OCR, o "
    "si no sabes si un área merece nodo propio, PROPÓN Y MARCA. No adivines en silencio.\n"
    "\n"
    "FORMATO DE SALIDA: devuelve EXCLUSIVAMENTE un objeto JSON válido, sin texto alrededor, "
    "con esta forma:\n"
    '{"groups": [{"code_label": "<etiqueta tal cual>", '
    '"parent_code_label": "<etiqueta del grupo padre|null>", '
    '"job_category_ids": [<ids existentes>], '
    '"source_excerpt": "<línea(s) exactas del convenio>", "source_locator": "<p.N>", '
    '"confidence": <0..1>, '
    '"uncertainty": {"field": "structure|membership|area|label", "reason": "<por qué>"}}], '
    '"notes": "<qué no pudiste determinar, o cadena vacía>"}'
)

# One convenio's own text. Doc 51 (Hostelería Navarra) is 30 pages and doc 56
# (COEAS Álava) is 43, so the cap has to hold a whole convenio rather than a
# fixture-sized file — the group structure is usually stated in one article, but
# WHICH article varies, so the model needs the whole text to find it.
PROPOSE_GROUPS_TEXT_CAP = 180000

# A group tree is small (a handful of nodes with one excerpt each), so this needs
# nothing like SEGMENT_MAX_TOKENS. 8192 is ample and keeps the call non-streaming.
PROPOSE_GROUPS_MAX_TOKENS = 8192


def _group_category_block(convenio: ConvenioCandidate) -> str:
    """The CLOSED category set for one convenio, with `group_code` marked as the
    weak evidence it is (see the system prompt's category section)."""
    if not convenio.job_categories:
        return (
            "Categorías profesionales existentes: NINGUNA.\n"
            "  Este convenio no tiene ninguna categoría cargada, lo cual es normal y no es un "
            "problema: propón solo grupos y áreas, sin adjuntar categorías. La estructura de "
            "grupos es suficiente por sí sola."
        )

    lines = [
        "Categorías profesionales existentes (conjunto CERRADO — usa solo estos ids, "
        "nunca inventes ni crees ninguna):",
    ]
    for jc in convenio.job_categories:
        evidence = f"   [indicio group_code: {jc.group_code!r} — puede ser basura, verifícalo contra el texto]" if jc.group_code else ""
        lines.append(f"  - id={jc.id}: {jc.name}{evidence}")
    return "\n".join(lines)


def _build_propose_groups_prompt(
    convenio: ConvenioCandidate,
    pages_text: str,
    observed_group_labels: list[str],
) -> str:
    text = (pages_text or "").strip()
    truncated = len(text) > PROPOSE_GROUPS_TEXT_CAP
    if truncated:
        text = text[:PROPOSE_GROUPS_TEXT_CAP] + "\n…[texto truncado]"

    blocks = [
        f"CONVENIO: {convenio.name}"
        + (f" (nº {convenio.numero})" if convenio.numero else "")
        + (f" · territorio: {convenio.territory_name}" if convenio.territory_name else "")
        + (f" · sector: {convenio.sector_name}" if convenio.sector_name else ""),
        "",
        "TEXTO DEL CONVENIO (la única fuente de verdad — toda cita debe salir de aquí):",
        text,
        "",
        _group_category_block(convenio),
    ]

    if observed_group_labels:
        blocks.append("")
        blocks.append(
            "Etiquetas de grupo que ya usan datos de referencia VERIFICADOS de este convenio "
            "(lista de comprobación, no fuente — tu estructura debería poder expresarlas todas):"
        )
        blocks.extend(f"  - {label}" for label in observed_group_labels)

    blocks.append("")
    blocks.append(
        "Propón la estructura de grupos de este convenio. Un área dentro de un grupo SOLO si el "
        "texto le da valores distintos, y siempre con su cita. Copia las etiquetas tal cual. "
        "No inventes categorías. Marca lo que no puedas determinar. Devuelve solo el JSON."
    )
    return "\n".join(blocks)


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


# --- OCR (Sprint 7e, ADR-0026) — vision transcription of a text-less page ----
# The system prompt below is the PRODUCTION COPY of the exact prompt scored in
# the engine/model eval (`hr-docs/sprints/sprint-07e/eval/engines/claude_vision.py`
# `SYSTEM_PROMPT`), including the Round-2 Adjustment 2 pinned table-placement
# contract (review.md §1.6/§2.2: a table page's title lives ONLY in
# `article_headers`; a footnote/plus-line block lives ONLY in one `columns` `es`
# entry; `table_rows` holds ONLY the grid — never left to the model's choice).
# Keep the two copies in lockstep: a prompt change here without a matching eval
# change (or vice versa) would silently invalidate ADR-0026's measured decision.
OCR_SYSTEM_PROMPT = (
    "Eres un transcriptor OCR. Tu única tarea es TRANSCRIBIR EXACTAMENTE el "
    "texto visible en la imagen de una página escaneada de un convenio "
    "colectivo español (a veces bilingüe euskera/castellano). NUNCA corrijas, "
    "completes, modernices, resumas ni \"limpies\" el texto — transcribe "
    "literalmente lo que ves, incluyendo erratas, mayúsculas y saltos de "
    "línea de artículo. Si una palabra es ilegible, escribe [ilegible] en su "
    "lugar en vez de inventarla.\n\n"
    "PASO 1 — determina el layout de la página:\n"
    '  - "two_column_bilingual": dos columnas verticales separadas por un '
    "gutter central, cada una en un idioma distinto (euskera / castellano).\n"
    '  - "two_column_monolingual": dos columnas verticales, ambas en '
    "castellano (layout tipo periódico, NO bilingüe).\n"
    '  - "single_column": una sola columna de prosa (aunque tenga un margen '
    "o índice lateral corto).\n"
    '  - "table": una tabla/rejilla salarial o anexo con filas y columnas de '
    "datos, sin prosa corrida.\n\n"
    "PASO 2 — transcribe según el layout:\n"
    '  - two_column_bilingual / two_column_monolingual: devuelve "columns" '
    "con DOS entradas, cada una con su \"order\" (0=izquierda, 1=derecha), "
    "su \"language\" (\"es\" o \"eu\" — para monolingüe ambas \"es\"), y su "
    "\"text\" completo en orden de lectura de arriba a abajo DENTRO de esa "
    "columna (nunca intercales texto de la otra columna).\n"
    '  - single_column: devuelve "columns" con UNA entrada, order=0, '
    "language=\"es\" o \"eu\".\n"
    '  - table: CONTRATO DE COLOCACIÓN FIJO para páginas de tabla (nunca lo '
    "dejes a tu criterio) — \"table_rows\" contiene EXCLUSIVAMENTE la rejilla "
    "de filas/columnas de datos (cabecera de columnas + filas de valores), "
    "una lista de listas de celdas, izquierda a derecha, arriba a abajo. Un "
    "título de tabla/anexo (p. ej. \"ANEXO I: TABLA SALARIAL...\") NUNCA es "
    "una fila de table_rows — va SOLO en \"article_headers\" (paso 3). Un "
    "pie de tabla o nota a pie (p. ej. \"Plus Festivo: 3,18 €/h.\", "
    "\"Kilometraje: 0,21 €/km.\") NUNCA es una fila de table_rows — va en "
    "\"columns\" como UNA entrada order=0, language=\"es\", con todas las "
    "líneas de nota unidas por salto de línea. Si no hay título o no hay "
    "notas al pie, simplemente omite esa parte (no inventes una entrada "
    "vacía). table_rows queda EXACTAMENTE del mismo tamaño que la rejilla "
    "visible — nunca una fila más por el título, nunca una fila más por "
    "una nota.\n\n"
    "PASO 3 — \"article_headers\": lista TODAS las cabeceras de artículo/"
    "capítulo/disposición que veas literalmente como aparecen (p. ej. "
    "\"Artículo 22\", \"22. artikulua\", \"CAPÍTULO V\", \"Disposición "
    "adicional primera\"), UNA por línea de cabecera, en el orden en que "
    "aparecen en la página. En una página \"table\", el TÍTULO de la tabla/"
    "anexo (p. ej. \"ANEXO I: TABLA SALARIAL DE 1-1-2025 A 31-12-2025\") es "
    "TAMBIÉN una cabecera y va aquí, no en table_rows.\n\n"
    "FORMATO DE SALIDA: devuelve EXCLUSIVAMENTE un objeto JSON válido, sin "
    "texto alrededor ni backticks:\n"
    '{"layout": "two_column_bilingual|two_column_monolingual|single_column|table", '
    '"columns": [{"order": 0, "language": "es|eu", "text": "..."}], '
    '"table_rows": [["cell", "cell"]], '
    '"article_headers": ["..."]}'
)

OCR_USER_PROMPT = (
    "Transcribe esta página escaneada siguiendo exactamente las instrucciones "
    "del sistema. Devuelve solo el JSON."
)

# Same table, same source, same date as the eval's copy (`eval/engines/
# claude_vision.py`'s `PRICING_PER_MTOK`) — checked 2026-09-07. Kept in
# lockstep: `cost_usd` here must match what the eval measured, or ADR-0026's
# "≈$3 for the whole backfill" projection silently drifts from reality.
OCR_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5": (3.00, 15.00),
    # Sprint 10-M: this entry was pre-staged at the introductory launch price
    # ($2.00/$10.00, in effect through 2026-08-31). Anthropic's standard rate
    # of $3.00/$15.00 took effect 2026-09-01 and was still current when this
    # sprint checked the live pricing page (2026-09-12) — i.e. Sonnet 5 is
    # priced identically to Sonnet 4.5 today, not cheaper. Corrected so
    # cost_usd is accurate from the first real Sonnet 5 call; re-check this
    # row if Anthropic revises the rate card again.
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    # Added Sprint 7g Item 1 (ADR-0029) — the ROUTER_MODEL, also used for the
    # escalation-explanation "Resumen IA" paragraph (hr-backend's
    # EscalationExplanationService, via `/explain` as of the Sprint 7g
    # fast-follow below — originally `/synthesise`, moved off it). Checked
    # against Anthropic's published rate card, 2026-09-10: $1.00 / $5.00 per MTok.
    "claude-haiku-4-5": (1.00, 5.00),
}
_OCR_DEFAULT_PRICING = (3.00, 15.00)

OCR_MAX_TOKENS = 4096

# Sprint 7g fast-follow (ADR-0029). EscalationExplanationService's "Resumen IA"
# paragraph originally reused `/synthesise` (SYSTEM_PROMPT, above) with the
# rendered facts posing as a single "chunk". Found live on staging: the model
# habitually appended a `[Fuente 1]`-style marker to EVERY sentence — an
# ingrained habit from that prompt's citation contract — which hr-backend's
# no-new-claims guard correctly rejected every time (3/3 live attempts across
# two reasons), so the AI paragraph never survived in practice. This is a
# DEDICATED prompt for a plain restatement task with none of that contract:
# no citation markers, no verbatim quoting, nothing beyond the supplied facts.
EXPLAIN_SYSTEM_PROMPT = (
    "Eres un/a redactor/a interno/a de Recursos Humanos. Tu ÚNICA tarea es "
    "reescribir una lista de hechos, ya verificados por otro sistema, como UN "
    "PÁRRAFO breve (3-5 frases) de prosa clara en español, dirigido a un/a "
    "compañero/a de RR.HH. que va a atender el caso.\n\n"
    "REGLAS ABSOLUTAS:\n"
    "1. Usa ÚNICAMENTE el contenido de los HECHOS proporcionados. No añadas "
    "ningún dato, cifra, nombre, fecha o afirmación que no esté literalmente "
    "en esos hechos. No completes huecos con tu conocimiento general.\n"
    "2. PROHIBIDO cualquier marcador o numeración de cita — nunca escribas "
    "'[Fuente N]', '(fuente)', 'según la fuente', ni ningún otro indicador de "
    "procedencia. Esta NO es una respuesta con citas: es la reescritura en "
    "prosa de una lista de hechos que ya te doy completa.\n"
    "3. PROHIBIDO usar comillas para citar los hechos literalmente. Reescribe "
    "con tus propias palabras, en prosa normal, sin comillas de ningún tipo.\n"
    "4. Un solo párrafo, sin markdown, sin listas, sin títulos, sin saludos.\n"
    "5. Responde en español.\n\n"
    "FORMATO DE SALIDA: devuelve EXCLUSIVAMENTE un objeto JSON válido, sin "
    "texto alrededor ni backticks, con esta forma exacta: "
    '{"answer": "<párrafo>"}'
)


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
            # Sprint 10-M: 1024 -> 4096 (matches /ground's first-tier budget).
            # Named, narrow exception to the model-swap-only fence, authorized
            # for the record: the 1024 ceiling was tuned against Sonnet 4.5's
            # completion style; keeping it would confound the model
            # measurement with a budget artifact (Sonnet 5 produced longer
            # completions on real chunk-dense questions and was observed
            # hitting this ceiling, cutting the JSON mid-structure ->
            # unparseable -> silent escalation). No retry-path added here —
            # /synthesise still has none, unlike /ground (see review.md
            # follow-up) — this only gives the first (only) attempt more room.
            max_tokens=4096,
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
        # Dedup key is null-safe (Sprint 7c Q7): a reference_fact source has
        # chunk_id=None, so it is keyed by its source_type+document_id instead of a
        # (colliding) None. Vector chunks key by chunk_id exactly as before.
        srckey_to_display: dict[object, int] = {}
        for n in cited_indices:
            try:
                i = int(n)
            except (TypeError, ValueError):
                continue
            if not (1 <= i <= len(chunks)):
                continue
            c = chunks[i - 1]
            src_key = c.chunk_id if c.chunk_id is not None else (getattr(c, "source_type", "chunk"), c.document_id)
            if src_key in srckey_to_display:
                # A second model-index pointing at an already-cited source: reuse
                # its display number so the marker still resolves 1:1.
                orig_to_display[i] = srckey_to_display[src_key]
                continue
            display = len(citations) + 1
            citations.append(
                {
                    "chunk_id": c.chunk_id,
                    "source_type": getattr(c, "source_type", "chunk"),
                    "document_id": c.document_id,
                    "page_from": c.page_from,
                    "page_to": c.page_to,
                    "authority_level": c.authority_level,
                }
            )
            orig_to_display[i] = display
            srckey_to_display[src_key] = display
            if c.authority_level:
                authority_used.add(c.authority_level)

        answer = _renumber_markers(answer, orig_to_display)

        top_score = max((c.score for c in chunks), default=0.0)
        grounded = len(citations) >= 1 and bool(answer)

        # Sprint 7g Item 1 (ADR-0029): cost_usd, same computation/table as the
        # other metered calls (propose-groups, ocr-page) — added here because
        # the escalation-explanation "Resumen IA" paragraph reuses THIS
        # endpoint with the cheap ROUTER_MODEL and the sprint spec requires
        # "cost logged per card". Purely additive: every existing caller of
        # /synthesise (the employee answer path) already ignores unknown
        # trace_fragment keys.
        in_tok = getattr(resp.usage, "input_tokens", None) or 0
        out_tok = getattr(resp.usage, "output_tokens", None) or 0
        price_in, price_out = OCR_PRICING_PER_MTOK.get(config.model, _OCR_DEFAULT_PRICING)
        cost_usd = round((in_tok / 1_000_000) * price_in + (out_tok / 1_000_000) * price_out, 6)

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
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "cost_usd": cost_usd,
                "synthesis_ms": elapsed_ms,
                "authority_used": authority_ordered,
            },
        )

    def explain(
        self,
        instruction: str,
        facts_text: str,
        api_key: str,
        config: ProviderConfig,
    ) -> ExplainResult:
        """Plain restatement of `facts_text` (Sprint 7g fast-follow, ADR-0029).
        Uses `EXPLAIN_SYSTEM_PROMPT` — a dedicated prompt with NO citation
        contract — never `SYSTEM_PROMPT` (that one is for `/synthesise`'s
        cited-answer task and is exactly what caused the `[Fuente N]` habit
        this call exists to avoid)."""
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, base_url=config.endpoint or None)
        user_prompt = f"Instrucción: {instruction}\n\nHECHOS:\n{facts_text}"

        started = time.monotonic()
        resp = client.messages.create(
            model=config.model,
            max_tokens=512,
            system=EXPLAIN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)

        raw_text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

        in_tok = getattr(resp.usage, "input_tokens", None) or 0
        out_tok = getattr(resp.usage, "output_tokens", None) or 0
        price_in, price_out = OCR_PRICING_PER_MTOK.get(config.model, _OCR_DEFAULT_PRICING)
        cost_usd = round((in_tok / 1_000_000) * price_in + (out_tok / 1_000_000) * price_out, 6)
        trace_fragment = {
            "provider": config.provider,
            "model": config.model,
            "explain_ms": elapsed_ms,
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "cost_usd": cost_usd,
        }

        try:
            envelope = _extract_json(raw_text)
        except (json.JSONDecodeError, ValueError):
            # Unparseable output → no paragraph. hr-backend's caller treats an
            # empty answer exactly like a failed no-new-claims check (falls
            # back to the deterministic sentences) — never guesses.
            return ExplainResult(answer="", trace_fragment={**trace_fragment, "parse_error": True})

        answer = str(envelope.get("answer", "")).strip()

        return ExplainResult(answer=answer, trace_fragment=trace_fragment)

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

    def propose_groups(
        self,
        convenio: ConvenioCandidate,
        pages_text: str,
        observed_group_labels: list[str],
        api_key: str,
        config: ProviderConfig,
    ) -> GroupProposalResult:
        """Read ONE convenio's text and propose its group tree (Sprint 7f,
        ADR-0028). Read-only and inert: hr-backend persists every node as
        `ai_agent`/`needs_review`, so a bad proposal costs a reviewer's click,
        never a wrong answer.

        Every structural invariant the DB enforces is ALSO enforced here, before
        hr-backend ever sees the payload, so a malformed tree degrades to fewer
        nodes rather than to a rejected batch: two levels only, no orphan
        sub-areas, a sub-area must cite its differing value, and category ids are
        validated against this convenio's closed set."""
        import anthropic  # lazy — dep only needed at call time

        client = anthropic.Anthropic(api_key=api_key, base_url=config.endpoint or None)
        user_prompt = _build_propose_groups_prompt(convenio, pages_text, observed_group_labels)

        started = time.monotonic()
        resp = client.messages.create(
            model=config.model,
            max_tokens=PROPOSE_GROUPS_MAX_TOKENS,
            system=PROPOSE_GROUPS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)

        raw_text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        in_tok = getattr(resp.usage, "input_tokens", None) or 0
        out_tok = getattr(resp.usage, "output_tokens", None) or 0
        price_in, price_out = OCR_PRICING_PER_MTOK.get(config.model, _OCR_DEFAULT_PRICING)
        cost_usd = round((in_tok / 1_000_000) * price_in + (out_tok / 1_000_000) * price_out, 6)

        base_trace = {
            "provider": config.provider,
            "model": config.model,
            "propose_ms": elapsed_ms,
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "cost_usd": cost_usd,
        }

        try:
            envelope = _extract_json(raw_text)
        except (json.JSONDecodeError, ValueError):
            # A group tree is small and this call is non-streaming, so a parse
            # failure is a genuine anomaly rather than the mid-array truncation
            # `_salvage_facts` exists for. Return nothing: the convenio simply
            # stays without a proposal, which is the safe state (the digit
            # matcher is untouched until Phase 3) and is visibly retryable.
            return GroupProposalResult(
                groups=[],
                trace_fragment={**base_trace, "parse_error": True, "group_count": 0},
            )

        raw_groups = envelope.get("groups") or []
        if not isinstance(raw_groups, list):
            raw_groups = []

        valid_category_ids = {jc.id for jc in convenio.job_categories}

        def label_key(value: object) -> str:
            """A dedupe key only. Real normalization is hr-backend's
            `GroupCodeNormalizer` — ONE implementation, and not the model's job
            (see the system prompt). This just collapses the trivial variance
            that would otherwise create two nodes for one printed label."""
            return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

        # Pass 1 — the roots, which are what a sub-area's parent must resolve to.
        root_keys: set[str] = set()
        for node in raw_groups:
            if isinstance(node, dict) and not node.get("parent_code_label"):
                if str(node.get("code_label", "")).strip():
                    root_keys.add(label_key(node.get("code_label")))

        # Pass 1b — SYNTHESISE a missing parent rather than dropping its areas.
        #
        # Found on the first live run, against Hostelería Navarra. The model read
        # article 19 correctly and said so in its notes, but represented the split
        # group by its two areas ALONE, without also emitting the group itself —
        # so both areas were orphans and both were dropped, leaving exactly the
        # UNDER-SPLIT the eval gates on. Dropping a cited split is the worst
        # available outcome: it is how someone entitled to 90 días gets told 60.
        #
        # Synthesising the parent invents nothing. Its printed label is already
        # given, verbatim, by the child's `parent_code_label`; the child carries
        # the excerpt that proves the group is split; and the node lands
        # `needs_review` like every other, flagged so the reviewer sees that this
        # one came from its children rather than from a line of its own.
        synthesised_parents = 0
        # Labels the model itself used for AREAS. A parent naming one of these is
        # a THIRD level, not a missing group, and synthesising it would defeat the
        # two-level limit by turning one area into both an area and a root.
        area_keys = {
            label_key(n.get("code_label"))
            for n in raw_groups
            if isinstance(n, dict) and n.get("parent_code_label") and str(n.get("code_label", "")).strip()
        }
        orphan_parents: dict[str, str] = {}
        for node in raw_groups:
            if not isinstance(node, dict):
                continue
            parent_raw = node.get("parent_code_label")
            parent_label = str(parent_raw).strip() if parent_raw else ""
            if not parent_label or not str(node.get("code_label", "")).strip():
                continue
            key = label_key(parent_label)
            if key in root_keys or key in area_keys:
                continue
            # Only for a CITED area — an uncited split is refused below, and a
            # refused split must not conjure a group on the way out.
            if str(node.get("source_excerpt", "")).strip():
                orphan_parents.setdefault(key, parent_label)

        for key, parent_label in orphan_parents.items():
            child = next(
                (
                    n
                    for n in raw_groups
                    if isinstance(n, dict) and label_key(n.get("parent_code_label")) == key
                ),
                None,
            )
            raw_groups.append(
                {
                    "code_label": parent_label,
                    "parent_code_label": None,
                    "job_category_ids": [],
                    "source_excerpt": (child or {}).get("source_excerpt"),
                    "source_locator": (child or {}).get("source_locator"),
                    "confidence": (child or {}).get("confidence"),
                    "uncertainty": {
                        "field": "code_label",
                        "reason": (
                            "Nodo reconstruido: el modelo propuso sus áreas pero no el grupo "
                            "en sí. La etiqueta es la que citan sus áreas; confírmala contra "
                            "el texto del convenio."
                        ),
                    },
                }
            )
            root_keys.add(key)
            synthesised_parents += 1

        groups: list[dict] = []
        seen: set[tuple[str, str]] = set()
        dropped_orphan_areas = 0
        dropped_unsupported_areas = 0
        dropped_category_ids = 0
        flagged_missing_excerpt = 0

        for node in raw_groups:
            if not isinstance(node, dict):
                continue

            code_label = str(node.get("code_label", "")).strip()
            if not code_label:
                continue  # a node with no printed label cannot be reviewed or matched

            parent_raw = node.get("parent_code_label")
            parent_label = str(parent_raw).strip() if parent_raw else ""
            excerpt = str(node.get("source_excerpt", "")).strip()

            uncertainty = node.get("uncertainty")
            if not (isinstance(uncertainty, dict) and uncertainty.get("reason")):
                uncertainty = None

            if parent_label:
                parent_key = label_key(parent_label)
                if parent_key not in root_keys:
                    # An area whose group was never proposed. Promoting it to a
                    # root would invent a group the model didn't claim exists.
                    dropped_orphan_areas += 1
                    continue
                if parent_key == label_key(code_label):
                    dropped_orphan_areas += 1  # self-parent
                    continue
                if not excerpt:
                    # THE granularity guard, enforced and not merely requested: a
                    # split with no citation is the unsupported inference this
                    # sprint exists to prevent, and an unnecessary split makes
                    # Phase 3 demand a distinction the convenio never made.
                    dropped_unsupported_areas += 1
                    continue
            elif not excerpt:
                # A root with no citation is still reviewable (the reviewer can
                # find it), so keep it — but never let it look confident.
                flagged_missing_excerpt += 1
                uncertainty = uncertainty or {
                    "field": "label",
                    "reason": "El modelo no citó el texto que respalda este grupo.",
                }

            key = (label_key(code_label), label_key(parent_label))
            if key in seen:
                continue
            seen.add(key)

            category_ids: list[int] = []
            for cid in node.get("job_category_ids") or []:
                if isinstance(cid, int) and cid in valid_category_ids:
                    if cid not in category_ids:
                        category_ids.append(cid)
                else:
                    # Closed-set validation (ADR-0011 by construction): a
                    # hallucinated or foreign-convenio category id can NEVER
                    # reach hr-backend, and the AI never mints vocabulary.
                    dropped_category_ids += 1

            confidence = node.get("confidence")
            if not isinstance(confidence, (int, float)):
                confidence = None

            groups.append(
                {
                    "code_label": code_label,
                    "parent_code_label": parent_label or None,
                    "job_category_ids": category_ids,
                    "source_excerpt": excerpt or None,
                    "source_locator": str(node.get("source_locator", "")).strip() or None,
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                }
            )

        # Two levels, structurally: a root has no parent and every survivor's
        # parent is a root, so nothing deeper than group › area can exist.
        return GroupProposalResult(
            groups=groups,
            trace_fragment={
                **base_trace,
                "group_count": sum(1 for g in groups if g["parent_code_label"] is None),
                "sub_area_count": sum(1 for g in groups if g["parent_code_label"] is not None),
                "dropped_orphan_areas": dropped_orphan_areas,
                "synthesised_parents": synthesised_parents,
                "dropped_unsupported_areas": dropped_unsupported_areas,
                "dropped_category_ids": dropped_category_ids,
                "flagged_missing_excerpt": flagged_missing_excerpt,
                "text_truncated": len((pages_text or "").strip()) > PROPOSE_GROUPS_TEXT_CAP,
                "notes": str(envelope.get("notes", "")).strip() or None,
            },
        )

    def ocr_page(
        self,
        image_bytes: bytes,
        api_key: str,
        config: ProviderConfig,
    ) -> OcrPageResult:
        """OCR one already-rendered page image (Sprint 7e, ADR-0026). Literal
        transcription only (no cleanup — see `OCR_SYSTEM_PROMPT`), bound to the
        pinned table-placement contract. On a parse failure returns
        `layout="parse_error"` with everything else empty so the caller (`app/
        ocr.py`) can surface a `provider_error` and leave the page `ocr_pending`
        for a retry — the same conservative shape every other parse-failure
        branch in this file already uses, never a guess at page content."""
        import base64

        import anthropic  # lazy — dep only needed at call time

        client = anthropic.Anthropic(api_key=api_key, base_url=config.endpoint or None)
        image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")

        started = time.monotonic()
        resp = client.messages.create(
            model=config.model,
            max_tokens=OCR_MAX_TOKENS,
            system=OCR_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
                        },
                        {"type": "text", "text": OCR_USER_PROMPT},
                    ],
                }
            ],
        )
        elapsed_s = time.monotonic() - started

        raw_text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        in_tok = getattr(resp.usage, "input_tokens", 0) or 0
        out_tok = getattr(resp.usage, "output_tokens", 0) or 0
        price_in, price_out = OCR_PRICING_PER_MTOK.get(config.model, _OCR_DEFAULT_PRICING)
        cost_usd = round((in_tok / 1_000_000) * price_in + (out_tok / 1_000_000) * price_out, 6)
        trace_fragment = {
            "provider": config.provider,
            "model": config.model,
            "sec_per_page": round(elapsed_s, 3),
            "cost_usd": cost_usd,
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
        }

        try:
            envelope = _extract_json(raw_text)
        except (json.JSONDecodeError, ValueError):
            return OcrPageResult(
                layout="parse_error",
                trace_fragment={**trace_fragment, "parse_error": True},
            )

        layout = str(envelope.get("layout", "single_column")).strip() or "single_column"
        columns = [c for c in (envelope.get("columns") or []) if isinstance(c, dict)]
        table_rows = [r for r in (envelope.get("table_rows") or []) if isinstance(r, list)]
        article_headers = [str(h) for h in (envelope.get("article_headers") or [])]

        # Bilingual iff a genuine two-column layout AND the two columns' own
        # reported languages differ — the same test extract_columns.py's native
        # path applies (one column reads as `eu`, the other `es`), just reading
        # the model's self-reported `language` instead of re-deriving it from
        # `_es_ratio` (the model already did that classification in PASO 2).
        languages = {str(c.get("language", "")).strip() for c in columns}
        bilingual = layout.startswith("two_column") and len(languages) > 1

        return OcrPageResult(
            layout=layout,
            columns=columns,
            table_rows=table_rows,
            article_headers=article_headers,
            bilingual=bilingual,
            trace_fragment=trace_fragment,
        )

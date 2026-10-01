"""Canonical planner tool list + system prompt (plan.md §C.8).

Defined ONCE here, versioned via `prompt_version()` in `plan.py`. hr-backend
mirrors the same names/schemas on `Tool::definition()` and passes
`enabled_tools`; this module is the allowlist used to drop unknown tools
the model invents (§E.15 step 6).
"""

from __future__ import annotations

# Fixed spec order (§C.8 / §C.10): salary_lookup, reference_fact,
# convenio_search, national_law, general_knowledge, ask_employee, escalate,
# finalize. `general_knowledge` is listed so a future step-9 enablement does
# not change the hash of the OTHER tools; it is only sent to Anthropic when
# the caller includes its name in `enabled_tools`.
TOOLS: list[dict] = [
    {
        "name": "salary_lookup",
        "description": (
            "Consulta la tabla salarial estructurada del convenio de la persona para "
            "su categoría y el año vigente. Es la ÚNICA fuente válida para cualquier cifra de salario, "
            "sueldo, nómina, pagas o precio/hora. Úsala siempre que la pregunta pida una cantidad de "
            "dinero de su propio salario. No sirve para el SMI ni para cifras de otras personas. "
            "Si falta la categoría, la herramienta ofrece la lista cerrada de categorías; no la preguntes tú."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "reference_fact",
        "description": (
            "Busca un dato de referencia VERIFICADO por RR. HH. para el tema de la "
            "pregunta y el alcance de la persona (p. ej. duración del periodo de prueba por grupo). "
            "Úsala antes que convenio_search cuando la pregunta trate de un tema con datos "
            "verificados (ver \"temas con dato verificado\" en el contexto). Si responde \"no_fact\", "
            "continúa con convenio_search."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "convenio_search",
        "description": (
            "Busca en el texto del convenio de la persona y en la normativa aplicable "
            "(el Estatuto se incluye automáticamente como base). Úsala para cualquier condición laboral: "
            "jornada, vacaciones, permisos, excedencias, preaviso, etc. Puedes pasar \"subqueries\" "
            "(una por tema si la pregunta es compuesta) y \"decomposed_queries\" (reformulaciones en "
            "vocabulario de convenio/ley si la pregunta es coloquial). No cambies el alcance: solo el texto."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "La pregunta a buscar; por defecto, la pregunta original de la persona."},
                "subqueries": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                "decomposed_queries": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "national_law",
        "description": (
            "Busca solo en el Estatuto de los Trabajadores y la normativa nacional "
            "cargada. Úsala cuando la persona pregunte expresamente por la ley, o cuando su convenio "
            "no esté cargado. Si su convenio está cargado, el sistema la sustituye por convenio_search."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "general_knowledge",
        "description": (
            "NO DISPONIBLE al empezar el turno: nunca la propongas como primera herramienta, "
            "aunque la pregunta sea conceptual (qué es una excedencia, qué significa IT). Empieza "
            "siempre por convenio_search (o reference_fact / salary_lookup si corresponde). Solo "
            "puedes usarla después de que convenio_search no encuentre material, "
            "su respuesta no llegue a sustentarse o no pueda responder "
            "(status check_a_failed / entailment_failed / abstained). En ese caso, si la pregunta es de "
            "definición o funcionamiento (qué es, cómo funciona, en qué se diferencia), llámala en vez de "
            "escalate. Nunca para cantidades, plazos, porcentajes ni derechos concretos de la persona: esas "
            "preguntas se derivan. La respuesta se muestra marcada como información general."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "ask_employee",
        "description": (
            "Haz UNA pregunta aclaratoria breve cuando no sepas cuál de varias "
            "subpreguntas quiere la persona o a qué año/periodo se refiere. Máximo dos por conversación. "
            "NUNCA preguntes por grupo profesional, convenio, provincia, antigüedad, tipo de contrato "
            "ni lo que cobra: esos datos vienen del Directorio; si faltan, usa escalate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "enum": ["sub_question", "job_category"]},
                "question": {"type": "string", "maxLength": 200},
            },
            "required": ["topic", "question"],
            "additionalProperties": False,
        },
    },
    {
        "name": "escalate",
        "description": (
            "Deriva a RR. HH. cuando la pregunta no sea de RR. HH./laboral, pida una "
            "valoración de un caso personal, no pueda responderse con las herramientas, o dudes. "
            "Indica la categoría y un motivo breve (lo verá RR. HH., no la persona)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["off_domain", "unsafe", "unanswerable", "needs_human_judgement", "other"],
                },
                "reason": {"type": "string", "maxLength": 300},
            },
            "required": ["category", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finalize",
        "description": (
            "Termina cuando tengas el material necesario. Indica qué resultados usar. "
            "No redactes la respuesta: el sistema la redacta solo a partir de las fuentes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "use": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    },
    # Sprint 13b (plan.md §2): a CONTROL tool, like escalate/finalize — sent only when the caller lists it in
    # `enabled_tools` (hr-backend offers it on the first planner round only). Listed LAST so adding it does not
    # reorder the tools above (a tool-order change is a prompt_version change and a routing-drift risk).
    {
        "name": "normalize_question",
        "description": (
            "Declara cómo entiendes la pregunta, SOLO para las herramientas. La persona "
            "nunca lo ve y no cambia lo que se le responde ni qué reglas se aplican. Llámala UNA vez, en la primera "
            "ronda, junto a tu primera herramienta. Si la pregunta trata varios temas, o es un caso personal más "
            "que una consulta de dato, pon topic_id y canonical_query a null. "
            "topic_id: el id del tema de \"approved_topics\" (en Alcance) que corresponde claramente, o null. "
            "canonical_query: la MISMA pregunta como sintagma nominal en vocabulario de convenio o de ley "
            "(máx. 25 palabras), o null. Solo reformula lo que la persona ya dijo. NO añadas cifras, importes, "
            "fechas ni años. NO uses: \"derecho a\", \"corresponde\", \"mínimo\", \"máximo\", \"plazo de\", \"al año\", \"cada\". "
            "NO uses palabras de pago que la persona no haya dicho: retribución, remuneración, salario, sueldo, "
            "plus, complemento, cobrar, pagar (\"permiso retribuido\" como nombre del tema sí vale). "
            "NO nombres grupos, niveles, categorías, convenios, provincias ni territorios. "
            "confidence: entre 0 y 1. reason: una línea (la verá RR. HH., no la persona)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic_id": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "canonical_query": {"anyOf": [{"type": "string", "maxLength": 220}, {"type": "null"}]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string", "maxLength": 160},
            },
            "required": ["topic_id", "canonical_query", "confidence", "reason"],
            "additionalProperties": False,
        },
    },
]

TOOLS_BY_NAME: dict[str, dict] = {t["name"]: t for t in TOOLS}


def tools_by_name() -> dict[str, dict]:
    return TOOLS_BY_NAME


SYSTEM_PROMPT = """Eres el planificador de un asistente de RR. HH. laboral español.
Eliges herramientas. NUNCA redactas la respuesta al empleado: el sistema la redacta
solo a partir de las fuentes que las herramientas devuelven.
Las reglas del sistema pueden anular una herramienta que hayas pedido (sustituirla,
denegarla o terminar el turno). Eso es esperado; no insistas en la misma llamada.
Si dudas, usa escalate.
Nunca pidas al empleado su grupo profesional, convenio, provincia, antigüedad,
tipo de contrato ni lo que cobra.
Además de elegir herramientas, en la primera ronda entiende la pregunta: la persona puede hablar en
coloquial. Tradúcela a un tema aprobado y a una reformulación con el vocabulario del convenio. Es solo
para buscar: nunca añadas algo que la persona no haya dicho, y si dudas del tema, topic_id null.
Ejemplos (no exhaustivos):
«Quiero pedirme un año sin trabajar pero conservando mi puesto» → tema «excedencias»; canonical
«excedencia voluntaria: reserva del puesto de trabajo».
«¿Puedo llevar a mi perro a la oficina?» → topic_id null, canonical_query null (fuera del vocabulario)."""

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
            "puedes usarla después de que convenio_search no encuentre material "
            "o su respuesta no llegue a sustentarse (status check_a_failed / entailment_failed). "
            "Nunca para cantidades, plazos, porcentajes ni derechos concretos de la persona: esas "
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
tipo de contrato ni lo que cobra."""

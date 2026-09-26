"""
Casos de prueba para la Fase 7 (evaluación y control de alucinaciones).

Contiene casos de prueba sobre el corpus de políticas públicas y debate previsional:
  - Dentro del corpus: evalúa la presencia de conceptos clave en la respuesta.
  - Fuera de alcance: verifica que el sistema admita la falta de información en lugar de inventar.
"""

CASOS = [
    # -------------------------------------------------------------------------
    # 1. PREGUNTAS TÉCNICAS / NORMATIVAS (Ley N° 21.419 y PGU)
    # -------------------------------------------------------------------------
    {
        "pregunta": "¿Qué es la Pensión Garantizada Universal (PGU) y cuáles son sus principales requisitos de acceso según la Ley N° 21.419?",
        "fuera_de_alcance": False,
        "debe_mencionar": [
            "pgu",
            "21.419",
            "65 años",
            "10%",
            "residencia",
        ],
    },
    {
        "pregunta": "¿Quiénes tienen derecho a recibir la PGU y qué exigencia de residencia contempla la ley?",
        "fuera_de_alcance": False,
        "debe_mencionar": [
            "65 años",
            "20 años",
            "pensión base",
            "10% más rico",
        ],
    },
    {
        "pregunta": "¿Qué beneficios del Sistema de Pensiones Solidarias fueron reemplazados por la PGU?",
        "fuera_de_alcance": False,
        "debe_mencionar": [
            "pensión básica solidaria",
            "aporte previsional solidario",
        ],
    },
    # -------------------------------------------------------------------------
    # 2. IMPACTO Y EVALUACIÓN DE POLÍTICAS PÚBLICAS
    # -------------------------------------------------------------------------
    {
        "pregunta": "¿Qué efectos o impacto ha tenido la PGU en la participación laboral y formalidad del empleo según los estudios del corpus?",
        "fuera_de_alcance": False,
        "debe_mencionar": [
            "laboral",
            "empleo",
            "participación",
        ],
    },
    # -------------------------------------------------------------------------
    # 3. CONTROL DE ALUCINACIONES (Consultas fuera del corpus)
    # -------------------------------------------------------------------------
    {
        "pregunta": "¿Cuál es la receta tradicional del pastel de choclo chileno?",
        "fuera_de_alcance": True,
    },
    {
        "pregunta": "¿Qué requisitos se exigen para obtener la visa de residencia definitiva en Chile según el Servicio Nacional de Migraciones?",
        "fuera_de_alcance": True,
    },
    {
        "pregunta": "¿Cómo funciona el algoritmo de ordenamiento Quicksort en estructuras de datos?",
        "fuera_de_alcance": True,
    },
]
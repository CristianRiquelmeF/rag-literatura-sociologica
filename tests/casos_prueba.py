"""
Casos de prueba para la Fase 7 (evaluación y control de alucinaciones).

Edita la lista CASOS de abajo con preguntas reales sobre TU corpus. Hay dos
tipos de caso:

  - Pregunta que el corpus SÍ cubre: agrega "debe_mencionar" con 2-4
    palabras o frases clave que la respuesta debería incluir si el sistema
    recuperó y usó el contexto correcto. No hace falta que calcen palabra
    por palabra, pero sí deben ser lo bastante específicas como para que no
    aparezcan "por casualidad" en cualquier respuesta genérica.

  - Pregunta que el corpus NO cubre: usa "fuera_de_alcance": True en vez de
    "debe_mencionar". Esta es la prueba real de control de alucinaciones —
    confirma que el sistema admite que no sabe, en vez de inventar una
    respuesta con apariencia de autoridad.

Los dos ejemplos de abajo son de partida, basados en preguntas que ya
probaste a mano. Reemplázalos y agrega más — idealmente cubriendo varios
papers distintos de tu corpus, no solo uno.
"""

CASOS = [
    {
        "pregunta": "¿qué es el social listening y en qué se diferencia del rating tradicional?",
        "debe_mencionar": ["social listening", "rating"],
    },
    {
        "pregunta": "¿cuál es la receta tradicional del pastel de choclo chileno?",
        "fuera_de_alcance": True,
    },
]

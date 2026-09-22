"""
Fase 7 — Evaluación y control de alucinaciones.

Requiere Python 3.10+.

Qué resuelve este script, y por qué:

  Hasta ahora validaste el sistema "a ojo", pregunta por pregunta. Esta fase
  automatiza esa validación con un pequeño set de casos de prueba que TÚ
  defines (en tests/casos_prueba.py), basados en lo que sabes que dice tu
  propio corpus. Cada caso es uno de dos tipos:

    1. PREGUNTA DENTRO DEL CORPUS: además de la pregunta, defines palabras o
       frases clave que la respuesta DEBERÍA mencionar si el sistema
       recuperó y usó el contexto correcto. El script revisa si aparecen —
       no es un análisis semántico sofisticado, es una verificación léxica
       simple, pero alcanza para detectar una regresión evidente (por
       ejemplo, si cambias el top_k o el prompt y el sistema deja de traer
       lo que antes traía bien).
    2. PREGUNTA FUERA DE ALCANCE: algo que sabes que tu corpus NO cubre. Acá
       se prueba que el sistema efectivamente diga que no encontró
       información, en vez de inventar una respuesta plausible. Esta es la
       prueba de control de alucinaciones propiamente tal.

  El resultado es un resumen pasa/no pasa por caso. No reemplaza tu criterio
  leyendo las respuestas completas, pero da una forma repetible de volver a
  correr la misma batería de preguntas cada vez que cambies algo (el prompt,
  el top_k, el modelo) y ver de inmediato si rompiste algo que antes
  funcionaba bien — el mismo principio de una suite de tests en cualquier
  proyecto de software, aplicado a un sistema que no da la misma respuesta
  exacta dos veces.

Uso:
    python src/evaluate.py
"""

import sys

from google import genai
from sentence_transformers import SentenceTransformer

from config import BASE_DIR, EMBEDDING_MODEL_NAME, GEMINI_API_KEY
from generate import generar_respuesta
from retrieve import buscar, cargar_coleccion

sys.path.insert(0, str(BASE_DIR / "tests"))

# Frases que cuentan como "el sistema admitió que no sabe". Si tus respuestas
# de rechazo usan otra redacción, agrégala aquí.
FRASES_DE_RECHAZO = [
    "no encontré información",
    "no encontre informacion",
    "no tengo información",
    "no cuento con información",
    "no está en el corpus",
    "no está disponible en el contexto",
    "no dispongo de información",
]


def evaluar_caso(caso: dict, modelo: SentenceTransformer, coleccion, cliente: genai.Client) -> dict:
    pregunta = caso["pregunta"]
    fragmentos = buscar(pregunta, modelo, coleccion)
    respuesta = generar_respuesta(pregunta, fragmentos, cliente) if fragmentos else ""
    respuesta_normalizada = respuesta.lower()

    if caso.get("fuera_de_alcance"):
        aprobado = any(frase in respuesta_normalizada for frase in FRASES_DE_RECHAZO)
        detalle = "rechazó correctamente" if aprobado else "NO rechazó (posible alucinación)"
    else:
        esperadas = [t.lower() for t in caso.get("debe_mencionar", [])]
        faltantes = [t for t in esperadas if t not in respuesta_normalizada]
        aprobado = not faltantes
        detalle = "todas las claves presentes" if aprobado else f"faltaron: {faltantes}"

    return {"pregunta": pregunta, "aprobado": aprobado, "detalle": detalle, "respuesta": respuesta}


def main():
    if not GEMINI_API_KEY:
        raise SystemExit("No se encontró GEMINI_API_KEY. Revisa tu .env.")

    try:
        from casos_prueba import CASOS
    except ImportError as e:
        raise SystemExit(
            "No se encontró tests/casos_prueba.py con tus casos de prueba "
            "(ver la plantilla que te compartí)."
        ) from e

    print(f"Cargando modelo y colección... ({len(CASOS)} casos de prueba)\n")
    modelo = SentenceTransformer(EMBEDDING_MODEL_NAME)
    coleccion = cargar_coleccion()
    cliente = genai.Client(api_key=GEMINI_API_KEY)

    resultados = []
    for i, caso in enumerate(CASOS, start=1):
        print(f"[{i}/{len(CASOS)}] {caso['pregunta']}")
        resultado = evaluar_caso(caso, modelo, coleccion, cliente)
        resultados.append(resultado)
        estado = "PASA" if resultado["aprobado"] else "FALLA"
        print(f"  {estado} — {resultado['detalle']}\n")

    aprobados = sum(r["aprobado"] for r in resultados)
    print("-" * 70)
    print(f"Resumen: {aprobados}/{len(resultados)} casos aprobados.")

    if aprobados < len(resultados):
        print("\nCasos que fallaron:")
        for r in resultados:
            if not r["aprobado"]:
                print(f"  - {r['pregunta']}  ({r['detalle']})")


if __name__ == "__main__":
    main()

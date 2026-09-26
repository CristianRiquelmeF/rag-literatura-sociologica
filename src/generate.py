"""
Fase 6 — Generación aumentada: prompt + LLM + citas.

Requiere Python 3.10+.

Qué resuelve este script, y por qué:

  Conecta la recuperación (Fase 5) con un LLM generativo (Gemini) para
  producir una respuesta en lenguaje natural — pero con dos restricciones
  no negociables, que son la razón de ser de un RAG bien hecho y no solo
  "un chat con LLM":

    1. EL MODELO SOLO PUEDE USAR EL CONTEXTO RECUPERADO. El prompt le
       prohíbe explícitamente completar con conocimiento propio. Esto no
       elimina el riesgo de alucinación al 100% (ningún prompt lo hace),
       pero lo reduce drásticamente frente a no poner ninguna restricción.
    2. SI EL CONTEXTO NO ALCANZA, EL MODELO DEBE DECIRLO. En vez de
       "rellenar" una respuesta plausible cuando la recuperación no trajo
       nada útil, se le pide decir explícitamente que no encontró la
       información — preferible un "no sé" honesto a una respuesta
       inventada con apariencia de autoridad.

  Cada fragmento de contexto se numera ([1], [2]...) y se le pide al modelo
  citar ese número en cada afirmación. Al final, el script imprime la lista
  REAL de fuentes recuperadas (documento, autor, similitud) — así, incluso
  si el modelo se equivoca citando un número, tú siempre ves de dónde salió
  cada fragmento que se usó como contexto.

  La temperatura se deja baja (0.2): para una tarea de respuesta basada en
  hechos, menos "creatividad" del modelo es mejor, no peor.

Uso interactivo:
    python src/generate.py

Uso con una sola pregunta:
    python src/generate.py "¿qué es la sociología computacional?"
"""

import sys
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, GEMINI_API_KEY, GEMINI_MODEL_NAME
from retrieve import buscar, cargar_coleccion

SYSTEM_INSTRUCTION = """\
Eres un asistente de investigación que responde preguntas EXCLUSIVAMENTE a \
partir del contexto entregado, extraído de documentos de política pública \
(proyectos de ley, informes de comisión, debate legislativo y análisis \
especializado).

Reglas estrictas:
1. Responde solo con información que esté explícitamente en el contexto. \
No uses conocimiento propio ni completes con suposiciones.
2. Si el contexto no contiene información suficiente para responder, dilo \
explícitamente ("No encontré información sobre esto en el corpus \
indexado.") en vez de inventar una respuesta.
3. Cada afirmación debe ir acompañada de la referencia entre corchetes \
correspondiente (ej. [1], [2]), usando los números de las fuentes \
listadas en el contexto.
4. Si el contexto muestra posturas distintas o en disputa sobre el mismo \
punto (ej. entre oficialismo y oposición, o entre versiones sucesivas de \
un proyecto de ley), señala esa diversidad explícitamente en vez de \
presentar una sola postura como si fuera consenso.
5. Responde en español, de forma clara y directa, sin relleno innecesario.
"""


def construir_contexto(fragmentos: list[dict]) -> str:
    bloques = [
        f'[{i}] (Título: "{frag["titulo"]}", Autor: {frag["autor"]})\n{frag["texto"]}'
        for i, frag in enumerate(fragmentos, start=1)
    ]
    return "\n\n".join(bloques)


def construir_prompt(pregunta: str, fragmentos: list[dict]) -> str:
    return f"Contexto:\n{construir_contexto(fragmentos)}\n\nPregunta: {pregunta}"


def generar_respuesta(
    pregunta: str, fragmentos: list[dict], cliente: genai.Client, intentos: int = 3, espera_inicial: float = 2.0
) -> str:
    """
    Llama a Gemini, mostrando siempre el error real si algo falla.

    Se reintenta SOLO ante errores de servidor (ServerError, ej. 503 por
    alta demanda) — son los únicos genuinamente transitorios. Un error de
    cliente (ClientError: cuota agotada, clave inválida, modelo
    inexistente) no se arregla reintentando la misma petición tres veces;
    se informa de inmediato con el detalle real en vez de perder tiempo.
    """
    prompt = construir_prompt(pregunta, fragmentos)
    config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, temperature=0.2)

    ultimo_error: Exception | None = None
    for intento in range(1, intentos + 1):
        try:
            respuesta = cliente.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt, config=config)
            return respuesta.text
        except genai_errors.ClientError as e:
            raise RuntimeError(f"Gemini rechazó la solicitud (no es un problema transitorio): {e}") from e
        except Exception as e:
            ultimo_error = e
            if intento < intentos:
                espera = espera_inicial * (2 ** (intento - 1))
                print(
                    f"  (Gemini falló [{type(e).__name__}: {e}] — "
                    f"reintentando en {espera:.0f}s, intento {intento}/{intentos})"
                )
                time.sleep(espera)

    raise RuntimeError(
        f"Gemini no respondió después de {intentos} intentos. "
        f"Último error: {type(ultimo_error).__name__}: {ultimo_error}"
    ) from ultimo_error


def imprimir_fuentes(fragmentos: list[dict]) -> None:
    print("\nFuentes recuperadas:")
    for i, frag in enumerate(fragmentos, start=1):
        print(
            f"  [{i}] {frag['titulo']} — {frag['autor']} "
            f"(fragmento #{frag['posicion']}, similitud {frag['similitud']})"
        )


def main():
    if not GEMINI_API_KEY:
        raise SystemExit(
            "No se encontró GEMINI_API_KEY. Revisa que tu archivo .env tenga la clave "
            "(ver .env.example) y que estés corriendo esto desde la raíz del proyecto."
        )

    print("Cargando modelo de embeddings y colección (puede tardar unos segundos)...")
    modelo_embeddings = SentenceTransformer(EMBEDDING_MODEL_NAME)
    coleccion = cargar_coleccion()
    cliente_gemini = genai.Client(api_key=GEMINI_API_KEY)
    print(f"Listo. {coleccion.count()} fragmentos indexados.\n")

    def responder(pregunta: str) -> None:
        fragmentos = buscar(pregunta, modelo_embeddings, coleccion)
        if not fragmentos:
            print("No se recuperó ningún fragmento del corpus.")
            return
        try:
            respuesta = generar_respuesta(pregunta, fragmentos, cliente_gemini)
        except RuntimeError as e:
            print(f"\n{e}\n")
            return
        print(f"\nPregunta: {pregunta}\n{'-' * 70}")
        print(respuesta)
        imprimir_fuentes(fragmentos)

    if len(sys.argv) > 1:
        responder(" ".join(sys.argv[1:]))
        return

    print("Modo interactivo. Escribe una pregunta y presiona Enter ('salir' para terminar).\n")
    while True:
        try:
            pregunta = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not pregunta or pregunta.lower() in {"salir", "exit", "quit"}:
            break
        responder(pregunta)


if __name__ == "__main__":
    main()

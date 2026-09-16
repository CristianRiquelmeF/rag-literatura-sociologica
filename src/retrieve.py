"""
Fase 5 — Recuperación semántica.

Requiere Python 3.10+.

Qué resuelve este script, y por qué:

  Dada una pregunta en lenguaje natural, la codifica con el MISMO modelo de
  embeddings usado para indexar en la Fase 4, y busca en Chroma los
  fragmentos más parecidos semánticamente — por significado, no por
  coincidencia exacta de palabras.

  Este script SOLO recupera texto; todavía no genera ninguna respuesta con
  un LLM (eso es la Fase 6), y es deliberado que sea así: sirve para validar
  con tus propios ojos que la recuperación trae fragmentos relevantes ANTES
  de conectarla a un modelo generativo. Si la recuperación falla acá —trae
  fragmentos que no tienen nada que ver con la pregunta— ningún prompt
  ingenioso lo va a arreglar después; hay que corregir la recuperación
  misma (probar otro top_k, revisar el chunking, etc.).

  Detalle importante: así como en la Fase 4 los fragmentos se indexaron con
  el prefijo "passage: ", acá la pregunta se codifica con el prefijo
  "query: " — es la otra mitad del esquema asimétrico de E5. Usar el
  prefijo equivocado en cualquiera de los dos lados degrada la búsqueda.

Uso interactivo (recomendado para probar varias preguntas seguidas):
    python src/retrieve.py

Uso con una sola pregunta (por ejemplo, para probar rápido o desde otro script):
    python src/retrieve.py "¿qué es la sociología computacional?"
"""

import logging
import sys

import chromadb
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, EMBEDDING_MODEL_NAME, TOP_K_RESULTS, VECTOR_DB_DIR

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cargar_coleccion():
    cliente = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    try:
        return cliente.get_collection(COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"No se encontró la colección '{COLLECTION_NAME}' en {VECTOR_DB_DIR}. "
            "Corre primero python src/embed.py."
        ) from e


def buscar(pregunta: str, modelo: SentenceTransformer, coleccion, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """Devuelve los top_k fragmentos más parecidos semánticamente a la pregunta."""
    vector_pregunta = modelo.encode([f"query: {pregunta}"], normalize_embeddings=True)

    resultado = coleccion.query(
        query_embeddings=vector_pregunta.tolist(),
        n_results=top_k,
    )

    fragmentos = []
    for i in range(len(resultado["ids"][0])):
        distancia = resultado["distances"][0][i]
        fragmentos.append(
            {
                "chunk_id": resultado["ids"][0][i],
                "texto": resultado["documents"][0][i],
                # Con espacio "cosine", Chroma devuelve distancia = 1 - similitud_coseno.
                # Se invierte acá para mostrar un número más intuitivo: 1.0 = idéntico, 0.0 = sin relación.
                "similitud": round(1 - distancia, 3),
                **resultado["metadatas"][0][i],
            }
        )
    return fragmentos


def imprimir_resultados(pregunta: str, fragmentos: list[dict]) -> None:
    print(f"\nPregunta: {pregunta}")
    print("-" * 70)
    if not fragmentos:
        print("(sin resultados)")
        return
    for i, frag in enumerate(fragmentos, start=1):
        print(f"[{i}] similitud={frag['similitud']}  |  {frag['titulo']}  (fragmento #{frag['posicion']})")
        extracto = frag["texto"][:280] + ("..." if len(frag["texto"]) > 280 else "")
        print(f"    {extracto}")
        print()


def main():
    print("Cargando modelo de embeddings y colección (puede tardar unos segundos)...")
    modelo = SentenceTransformer(EMBEDDING_MODEL_NAME)
    coleccion = cargar_coleccion()
    print(f"Colección '{COLLECTION_NAME}' cargada: {coleccion.count()} fragmentos indexados.\n")

    # Modo de una sola pregunta: python src/retrieve.py "pregunta aca"
    if len(sys.argv) > 1:
        pregunta = " ".join(sys.argv[1:])
        imprimir_resultados(pregunta, buscar(pregunta, modelo, coleccion))
        return

    # Modo interactivo: pide preguntas por consola hasta que se escriba "salir".
    print("Modo interactivo. Escribe una pregunta y presiona Enter ('salir' para terminar).\n")
    while True:
        try:
            pregunta = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not pregunta or pregunta.lower() in {"salir", "exit", "quit"}:
            break
        imprimir_resultados(pregunta, buscar(pregunta, modelo, coleccion))


if __name__ == "__main__":
    main()

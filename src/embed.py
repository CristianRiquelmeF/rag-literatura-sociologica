"""
Fase 4 — Embeddings y carga a la base vectorial (Chroma).

Requiere Python 3.10+.

Qué resuelve este script, y por qué:

  1. EMBEDDINGS LOCALES, NO POR API. Se usa un modelo de sentence-transformers
     que corre en tu propia máquina (intfloat/multilingual-e5-base): no hay
     costo por fragmento indexado, no depende de un servicio externo, y el
     contenido de tus papers no sale de tu computador en esta etapa. La
     primera vez que corras esto, el modelo (~1.1 GB) se descarga una sola
     vez y queda cacheado localmente.
  2. EL MODELO E5 DISTINGUE "PREGUNTA" DE "PASAJE". A diferencia de un modelo
     de similitud genérico, E5 fue entrenado esperando el prefijo "query: "
     para preguntas y "passage: " para el texto que se indexa — usar el
     prefijo correcto en cada lado mejora notablemente la calidad de la
     recuperación (Fase 5). Acá, como estamos indexando el corpus, todos los
     fragmentos se codifican con el prefijo "passage: ".
  3. RECONSTRUCCIÓN COMPLETA EN CADA CORRIDA. Cada vez que corres este script
     se borra la colección anterior en Chroma y se recrea desde cero a partir
     de chunks.jsonl. Para un corpus chico esto es más simple y más seguro
     que actualizar de forma incremental: evita fragmentos duplicados o
     huérfanos si volviste a correr una fase anterior con cambios. Si el
     corpus creciera mucho (miles de fragmentos), esto es lo primero que se
     cambiaría por una actualización incremental (upsert por doc_id).

Uso:
    python src/embed.py
"""

import json
import logging

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHUNKS_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, VECTOR_DB_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cargar_fragmentos() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"No existe {CHUNKS_PATH}. Corre primero python src/chunk.py.")
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def main():
    fragmentos = cargar_fragmentos()
    if not fragmentos:
        logger.warning("chunks.jsonl está vacío. Nada que indexar.")
        return

    logger.info(
        "Cargando modelo de embeddings (%s)... la primera vez descarga ~1.1 GB, puede tardar.",
        EMBEDDING_MODEL_NAME,
    )
    modelo = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # E5 espera el prefijo "passage: " para el texto que se indexa
    # (ver punto 2 del docstring de este archivo).
    textos_con_prefijo = [f"passage: {frag['texto']}" for frag in fragmentos]

    logger.info("Generando embeddings para %d fragmentos...", len(fragmentos))
    embeddings = modelo.encode(
        textos_con_prefijo,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,  # vectores de norma 1 -> similitud coseno directa
    )

    logger.info("Conectando a Chroma en %s", VECTOR_DB_DIR)
    cliente = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    try:
        cliente.delete_collection(COLLECTION_NAME)
        logger.info("Colección previa '%s' eliminada — se reconstruye desde cero.", COLLECTION_NAME)
    except Exception:
        pass  # no existía todavía, no hay nada que borrar

    coleccion = cliente.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    coleccion.add(
        ids=[frag["chunk_id"] for frag in fragmentos],
        embeddings=embeddings.tolist(),
        documents=[frag["texto"] for frag in fragmentos],
        metadatas=[
            {
                "doc_id": frag["doc_id"],
                "titulo": frag["titulo"],
                "autor": frag["autor"],
                "posicion": frag["posicion"],
            }
            for frag in fragmentos
        ],
    )

    logger.info(
        "Listo. %d fragmentos indexados en la colección '%s' (%s).",
        len(fragmentos), COLLECTION_NAME, VECTOR_DB_DIR,
    )


if __name__ == "__main__":
    main()

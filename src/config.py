"""
Configuración central del proyecto.

Todas las rutas y parámetros del pipeline se definen aquí para no repetir
"números mágicos" ni rutas hardcodeadas en cada script. Cada fase importa
lo que necesita desde este módulo.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

# --- Rutas del proyecto ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_PATH = DATA_PROCESSED_DIR / "chunks.jsonl"
VECTOR_DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "papers_sociologia"

# --- Modelo de embeddings (local, multilingüe, gratuito) ---
# E5 usa prefijos "query: " y "passage: " para diferenciar el rol del texto
# que se está codificando — el detalle se explica y se usa en la Fase 4.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# --- LLM generativo (Fase 6) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-3.5-flash"

# --- Parámetros de chunking (se ajustan y justifican en la Fase 3) ---
CHUNK_SIZE = 800       # caracteres aproximados por fragmento
CHUNK_OVERLAP = 150    # solapamiento entre fragmentos consecutivos

# --- Parámetros de recuperación (se ajustan y justifican en la Fase 5) ---
TOP_K_RESULTS = 5      # cuántos fragmentos recuperar por pregunta

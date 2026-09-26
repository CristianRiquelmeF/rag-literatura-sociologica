"""
Fase 8 — Interfaz web (Streamlit).

Requiere Python 3.10+.

Qué resuelve este script, y por qué:

  Envuelve el pipeline completo (recuperación de la Fase 5 + generación de
  la Fase 6) en una interfaz de chat, para tener una demo funcional que se
  pueda mostrar en una entrevista sin depender de la terminal.

  Detalle importante de cómo funciona Streamlit: reejecuta TODO el script
  de arriba a abajo cada vez que el usuario interactúa con la página
  (escribe algo, aprieta un botón). Sin @st.cache_resource, eso significaría
  recargar el modelo de embeddings (~1.1 GB) y reconectar a Chroma en CADA
  pregunta — inviable. cache_resource asegura que el modelo y la conexión
  se cargan una sola vez por sesión, sin importar cuántas preguntas se
  hagan después.

Uso:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

# app.py vive en la raíz del proyecto, pero los módulos del pipeline están
# en src/ — se agrega esa carpeta al path para poder importarlos por su
# nombre simple, igual que se importan entre sí dentro de src/.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import EMBEDDING_MODEL_NAME, GEMINI_API_KEY  # noqa: E402
from generate import generar_respuesta  # noqa: E402
from retrieve import buscar, cargar_coleccion  # noqa: E402
from google import genai  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

st.set_page_config(page_title="SocioLit-RAG", page_icon="📚", layout="centered")


@st.cache_resource(show_spinner="Cargando modelo de embeddings y colección (puede tardar la primera vez)...")
def cargar_recursos():
    modelo = SentenceTransformer(EMBEDDING_MODEL_NAME)
    coleccion = cargar_coleccion()
    cliente = genai.Client(api_key=GEMINI_API_KEY)
    return modelo, coleccion, cliente


st.title("📋 Política-RAG | Observatorio de Políticas Públicas")
st.caption(
    "Asistente conversacional para el análisis de políticas públicas. "
    "Responde exclusivamente a partir del corpus indexado, citando las fuentes de cada afirmación."
)

if not GEMINI_API_KEY:
    st.error("No se encontró GEMINI_API_KEY. Revisa tu archivo .env.")
    st.stop()

try:
    modelo, coleccion, cliente = cargar_recursos()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

st.caption(f"{coleccion.count()} fragmentos indexados en el corpus.")

if "historial" not in st.session_state:
    st.session_state.historial = []


def mostrar_fuentes(fragmentos: list[dict]) -> None:
    with st.expander(f"Fuentes ({len(fragmentos)})"):
        for i, f in enumerate(fragmentos, start=1):
            st.markdown(f"**[{i}] {f['titulo']}** — {f['autor']} (similitud {f['similitud']})")


# Redibuja las preguntas y respuestas anteriores de esta sesión.
for pregunta_previa, respuesta_previa, fuentes_previas in st.session_state.historial:
    with st.chat_message("user"):
        st.markdown(pregunta_previa)
    with st.chat_message("assistant"):
        st.markdown(respuesta_previa)
        if fuentes_previas:
            mostrar_fuentes(fuentes_previas)

pregunta = st.chat_input("Escribe tu pregunta sobre el corpus...")

if pregunta:
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en el corpus y generando la respuesta..."):
            fragmentos = buscar(pregunta, modelo, coleccion)
            if not fragmentos:
                respuesta = "No se recuperó ningún fragmento del corpus para esta pregunta."
            else:
                try:
                    respuesta = generar_respuesta(pregunta, fragmentos, cliente)
                except RuntimeError as e:
                    respuesta = str(e)

        st.markdown(respuesta)
        if fragmentos:
            mostrar_fuentes(fragmentos)

    st.session_state.historial.append((pregunta, respuesta, fragmentos))

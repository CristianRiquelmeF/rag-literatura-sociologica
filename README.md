# SocioLit-RAG — Asistente conversacional sobre literatura sociológica

Sistema de Retrieval-Augmented Generation (RAG) construido desde cero para consultar,
en lenguaje natural, un corpus propio de papers y ensayos de sociología y ciencias
sociales computacionales. Cada respuesta cita la fuente exacta (documento y fragmento)
de donde salió — nunca responde "de memoria".

## Motivación

Este proyecto nace de una necesidad doble:

1. **Uso práctico**: un asistente propio para consultar rápido la literatura que
   se viene leyendo (sociología computacional, IA aplicada a investigación social,
   análisis de discurso, Critical Algorithm Studies, STS aplicado a IA).
2. **Portafolio**: demostrar manejo práctico de IA generativa — LLMs vía API,
   prompt engineering, arquitectura RAG, control de alucinaciones — más allá del
   machine learning clásico y el NLP con fine-tuning que ya se cubren en otros
   proyectos del portafolio.

## Por qué RAG y no "pegarle los PDFs a un chat"

Pegar el texto completo de varios papers en cada conversación con un LLM tiene
tres problemas: (1) el contexto se llena rápido y sale caro en tokens en cada
pregunta, (2) no hay manera de saber de qué documento exacto salió una afirmación,
y (3) el modelo puede "completar" con conocimiento propio en vez de ceñirse a los
documentos (alucinación). El RAG resuelve los tres: indexa el corpus una sola vez,
recupera solo los fragmentos relevantes para cada pregunta, y obliga al modelo a
responder basado en esos fragmentos, citándolos.

## Arquitectura

El sistema tiene dos pipelines independientes (ver diagramas en la conversación):

**Pipeline de indexación** (offline, se corre una vez por documento nuevo):
PDFs → limpieza de texto → chunking → embeddings (modelo local) → base vectorial (Chroma)

**Pipeline de consulta** (online, se corre en cada pregunta del usuario):
Pregunta → recuperación semántica (top-k) → generación aumentada con LLM (Gemini) → respuesta citada

## Hoja de ruta

- [x] **Fase 1** — Alcance, entorno y estructura del proyecto
- [x] **Fase 2** — Ingesta y limpieza de PDFs académicos (probada con PDFs sintéticos; falta validar con papers reales)
- [ ] **Fase 3** — Chunking del texto
- [ ] **Fase 4** — Embeddings y base vectorial (Chroma)
- [ ] **Fase 5** — Recuperación semántica
- [ ] **Fase 6** — Generación aumentada (prompt + LLM + citas)
- [ ] **Fase 7** — Evaluación y control de alucinaciones
- [ ] **Fase 8** — Interfaz (Streamlit) y despliegue
- [ ] **Fase 9** — Documentación final para portafolio/entrevista

## Decisiones de diseño (para poder defenderlas en una entrevista)

- **Embeddings locales (E5 multilingüe) en vez de una API de embeddings**: no
  depende de un servicio externo, no tiene costo por documento indexado, y evita
  enviar el contenido íntegro de tus papers a un tercero — un argumento de
  privacidad de datos de investigación defendible en cualquier entrevista.
- **Gemini solo para la generación, no para los embeddings**: separa
  responsabilidades. El modelo generativo entra únicamente en la última etapa,
  así que su costo se limita a las preguntas efectivamente hechas, no a indexar
  el corpus completo.
- **Chroma como base vectorial**: no requiere servidor, corre localmente en un
  archivo, y es la opción estándar para un proyecto de este tamaño (decenas a un
  par de miles de fragmentos). Migrar a una base vectorial gestionada (Pinecone,
  Weaviate Cloud) sería un cambio de una sola pieza si el proyecto creciera.
- **Construcción desde cero en vez de un framework como LangChain**: el objetivo
  es entender y poder explicar cada pieza del pipeline sin que quede oculta
  detrás de abstracciones. Una vez que el sistema funcione de punta a punta,
  migrar a LangChain/LlamaIndex es un ejercicio de refactor, no de diseño.

## Instalación

```bash
git clone <tu-repo>
cd sociolit-rag
python -m venv venv
source venv/bin/activate       # En Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edita .env y pega tu clave gratuita de https://aistudio.google.com/apikey
```

## Estructura del proyecto

```
sociolit-rag/
├── data/
│   ├── raw/          # PDFs originales (NO se sube a git — ver nota de derechos de autor)
│   └── processed/    # Texto limpio extraído, listo para trocear
├── src/
│   ├── config.py     # Configuración central (rutas, modelos, parámetros)
│   ├── ingest.py     # Fase 2 — extracción y limpieza de PDFs
│   ├── chunk.py      # Fase 3 — troceo de texto
│   ├── embed.py      # Fase 4 — embeddings y carga a Chroma
│   ├── retrieve.py   # Fase 5 — recuperación semántica
│   └── generate.py   # Fase 6 — prompt + LLM + citas
├── notebooks/        # Exploración y validación manual de cada fase
├── tests/            # Preguntas de prueba y evaluación (Fase 7)
├── app.py            # Interfaz Streamlit (Fase 8)
├── requirements.txt
├── .env.example
└── README.md
```

## Nota sobre derechos de autor

Los PDFs de papers académicos casi siempre tienen copyright. `data/raw/` está en
`.gitignore` a propósito: el repositorio público en GitHub debe contener el
pipeline (el código), no los documentos. Si quieres mostrar el proyecto
funcionando ante un tercero sin exponer texto con copyright, dos opciones:
(a) correr la demo en vivo con tu propia carpeta local durante la entrevista, o
(b) armar un corpus de prueba con papers de acceso abierto (SciELO, Redalyc,
repositorios institucionales) que sí puedas incluir o referenciar libremente.

## Estado actual

Fase 1 y 2 completas. `src/ingest.py` fue probado con PDFs sintéticos que
simulan los problemas reales de un paper académico: layout a dos columnas,
encabezado de revista repetido en cada hoja, números de página que cambian
por hoja, una palabra cortada por guion de fin de línea, metadata de título
genérica ("untitled") y un PDF sin texto extraíble (simulando un escaneo).
Los cinco casos se manejaron correctamente.

**Pendiente antes de pasar a la Fase 3**: correr `python src/ingest.py` con
tus 5-10 papers reales (cópialos a `data/raw/` primero), abrir los `.json`
generados en `data/processed/` y confirmar que:
1. el texto queda en el orden de lectura correcto (no mezclado entre columnas),
2. el título detectado por documento es razonable,
3. no quedaron restos de encabezado/pie de página.

Si algo no calza con tus PDFs reales (typesetting distinto, columnas de ancho
muy desigual, etc.), es el momento de ajustarlo antes de construir el
chunking sobre un texto mal extraído.

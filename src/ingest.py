"""
Fase 2 — Ingesta y limpieza de PDFs académicos.

Requiere Python 3.10+ (usa sintaxis de tipos moderna: `X | None`, `list[dict]`).

Qué resuelve este script, y por qué hace falta resolverlo:

  1. ORDEN DE LECTURA EN PAPERS A DOS COLUMNAS. La mayoría de los lectores de
     PDF (incluida la extracción "ingenua" de pypdf/pdfplumber) recorren la
     página de arriba hacia abajo por coordenada Y, sin saber que un paper
     científico suele tener dos columnas. El resultado es que el texto queda
     intercalado: una línea de la columna izquierda, la línea de la columna
     derecha que está a la misma altura, otra vez la izquierda... Esto rompe
     por completo la coherencia semántica de cada fragmento cuando lleguemos
     a la Fase 3 (chunking). Este script detecta el "corredor" en blanco entre
     columnas y extrae cada columna completa, en orden, antes de pasar a la
     siguiente.
  2. ENCABEZADOS Y PIES DE PÁGINA REPETIDOS. El nombre de la revista, el DOI
     o el número de página que se repiten en cada hoja no aportan nada al
     contenido y contaminan los fragmentos. Se detectan por frecuencia y se
     eliminan.
  3. PALABRAS CORTADAS POR GUION DE FIN DE LÍNEA ("sociolo-\\ngía"). Se
     recomponen para no dejar fragmentos con palabras rotas.
  4. METADATA BÁSICA (título, autor) para poder citar la fuente después.

Qué NO resuelve (fuera del alcance de esta fase, a propósito):
  - PDFs escaneados sin capa de texto (imágenes puras): se detectan y se
    omiten con una advertencia. Si tienes papers así, pásalos por OCR
    (ej. Tesseract) antes de ponerlos en data/raw/.
  - Tablas y figuras: se extraen como texto plano si tienen texto seleccionable,
    pero no se reconstruye su estructura tabular.

Uso:
    python src/ingest.py
"""

import gc
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pdfplumber

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DocumentoProcesado:
    doc_id: str
    archivo_origen: str
    titulo: str
    autor: str
    num_paginas: int
    texto: str
    num_caracteres: int


def detectar_columnas(palabras: list[dict], ancho_pagina: float) -> list[list[dict]]:
    ANCHO_BIN = 5
    ANCHO_MINIMO_CORREDOR = 15

    num_bins = int(ancho_pagina // ANCHO_BIN) + 1
    cobertura = [0] * num_bins

    for palabra in palabras:
        bin_inicio = int(palabra["x0"] // ANCHO_BIN)
        bin_fin = int(palabra["x1"] // ANCHO_BIN)
        for b in range(bin_inicio, min(bin_fin + 1, num_bins)):
            cobertura[b] += 1

    tercio_izq, tercio_der = num_bins // 3, 2 * num_bins // 3
    bins_minimos = max(int(ANCHO_MINIMO_CORREDOR // ANCHO_BIN), 1)

    inicio_corredor, racha = None, 0
    for b in range(tercio_izq, tercio_der):
        if cobertura[b] == 0:
            racha += 1
            if racha >= bins_minimos:
                inicio_corredor = b - racha + 1
        else:
            racha = 0

    if inicio_corredor is None:
        return [sorted(palabras, key=lambda p: (round(p["top"]), p["x0"]))]

    x_gutter = (inicio_corredor + bins_minimos / 2) * ANCHO_BIN
    columna_izq, columna_der = [], []
    for p in palabras:
        centro = (p["x0"] + p["x1"]) / 2
        (columna_izq if centro < x_gutter else columna_der).append(p)

    columna_izq.sort(key=lambda p: (round(p["top"]), p["x0"]))
    columna_der.sort(key=lambda p: (round(p["top"]), p["x0"]))
    return [columna_izq, columna_der]


def extraer_texto_pagina(pagina) -> str:
    palabras = pagina.extract_words(use_text_flow=False, keep_blank_chars=False)
    if not palabras:
        return ""

    grupos = detectar_columnas(palabras, pagina.width)

    lineas = []
    for grupo in grupos:
        linea_actual, top_actual = [], None
        for palabra in grupo:
            if top_actual is None or abs(palabra["top"] - top_actual) > 3:
                if linea_actual:
                    lineas.append(" ".join(linea_actual))
                linea_actual, top_actual = [palabra["text"]], palabra["top"]
            else:
                linea_actual.append(palabra["text"])
        if linea_actual:
            lineas.append(" ".join(linea_actual))

    return "\n".join(lineas)


def quitar_guiones_de_corte(texto: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", texto)


def quitar_encabezados_repetidos(paginas_texto: list[str]) -> list[str]:
    if len(paginas_texto) < 3:
        return paginas_texto

    primeras = [p.split("\n")[0] for p in paginas_texto if p]
    ultimas = [p.split("\n")[-1] for p in paginas_texto if p]
    conteo = Counter(primeras + ultimas)
    umbral = len(paginas_texto) * 0.5
    lineas_a_quitar = {linea for linea, freq in conteo.items() if freq >= umbral and linea.strip()}

    if not lineas_a_quitar:
        return paginas_texto

    logger.info("Encabezados/pies repetidos detectados y removidos: %s", lineas_a_quitar)
    return [
        "\n".join(l for l in pagina.split("\n") if l not in lineas_a_quitar)
        for pagina in paginas_texto
    ]


_PATRON_NUM_PAGINA = re.compile(r"^\s*(p[aá]gina|page)?\s*\d{1,4}\s*(/\s*\d{1,4})?\s*$", re.IGNORECASE)


def quitar_numeros_de_pagina(paginas_texto: list[str]) -> list[str]:
    return [
        "\n".join(l for l in pagina.split("\n") if not _PATRON_NUM_PAGINA.match(l.strip()))
        for pagina in paginas_texto
    ]


_TITULOS_GENERICOS = {"untitled", "unknown", "sin titulo", "sin título", "documento", "document", "pdf"}
_AUTORES_GENERICOS = {"anonymous", "unknown", ""}


def extraer_metadata(pdf, ruta_archivo: Path, texto_primera_pagina: str) -> tuple[str, str]:
    meta = pdf.metadata or {}
    titulo = (meta.get("Title") or "").strip()
    autor = (meta.get("Author") or "").strip()

    if not titulo or titulo.lower() in _TITULOS_GENERICOS:
        titulo = ""
        for linea in texto_primera_pagina.split("\n"):
            if 10 < len(linea.strip()) < 200:
                titulo = linea.strip()
                break
        if not titulo:
            titulo = ruta_archivo.stem

    if autor.lower() in _AUTORES_GENERICOS:
        autor = "Autor no disponible"

    return titulo, autor


def procesar_pdf(ruta_archivo: Path) -> DocumentoProcesado | None:
    logger.info("Procesando: %s", ruta_archivo.name)

    try:
        paginas_limpias = []
        num_paginas = 0

        with pdfplumber.open(ruta_archivo) as pdf:
            num_paginas = len(pdf.pages)

            for i, pagina in enumerate(pdf.pages, start=1):
                texto = extraer_texto_pagina(pagina)
                if texto.strip():
                    texto = quitar_numeros_de_pagina([texto])[0]
                    paginas_limpias.append(texto)

                if i % 30 == 0:
                    gc.collect()

            if len(paginas_limpias) < 3 or sum(len(p) for p in paginas_limpias) < 500:
                logger.warning(
                    "%s tiene muy poco texto extraíble — probablemente es un PDF "
                    "escaneado sin OCR. Se omite.",
                    ruta_archivo.name,
                )
                return None

            paginas_limpias = quitar_encabezados_repetidos(paginas_limpias)
            texto_final = "\n".join(paginas_limpias)
            texto_final = quitar_guiones_de_corte(texto_final)
            texto_final = re.sub(r"\n{3,}", "\n\n", texto_final).strip()

            titulo, autor = extraer_metadata(pdf, ruta_archivo, texto_final[:3000])

            del paginas_limpias
            gc.collect()

            return DocumentoProcesado(
                doc_id=ruta_archivo.stem,
                archivo_origen=ruta_archivo.name,
                titulo=titulo,
                autor=autor,
                num_paginas=num_paginas,
                texto=texto_final,
                num_caracteres=len(texto_final),
            )

    except Exception:
        logger.exception("Falló el procesamiento de %s", ruta_archivo.name)
        return None


def main():
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(DATA_RAW_DIR.glob("*.pdf"))

    if not pdfs:
        logger.warning("No hay PDFs en %s. Copia tus papers ahí antes de correr esto.", DATA_RAW_DIR)
        return

    logger.info("Encontrados %d PDFs para procesar.", len(pdfs))
    procesados = omitidos = saltados = 0

    for ruta in pdfs:
        salida = DATA_PROCESSED_DIR / f"{ruta.stem}.json"

        # ←←← LÓGICA DE SALTO AÑADIDA
        if salida.exists():
            logger.info("Saltando %s (ya procesado previamente)", ruta.name)
            saltados += 1
            continue

        doc = procesar_pdf(ruta)
        if doc is None:
            omitidos += 1
            continue

        salida.write_text(json.dumps(asdict(doc), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "  -> %s | %d páginas | %d caracteres | título detectado: %s",
            salida.name, doc.num_paginas, doc.num_caracteres, doc.titulo[:60],
        )
        procesados += 1

    logger.info("Listo. Procesados: %d | Saltados: %d | Omitidos: %d", procesados, saltados, omitidos)


if __name__ == "__main__":
    main()

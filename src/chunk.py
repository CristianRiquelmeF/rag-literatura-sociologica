"""
Fase 3 — Chunking del texto procesado en la Fase 2.

Requiere Python 3.10+.

Qué resuelve este script, y por qué:

  Dividir el texto en fragmentos ("chunks") del tamaño adecuado para generar
  embeddings y recuperarlos después. Chunking NO es solo cortar cada N
  caracteres: cortar a mitad de una oración destruye el significado del
  fragmento y degrada tanto el embedding como la respuesta final del LLM.
  Por eso el proceso es:

    1. El texto se separa en oraciones completas (NLTK, tokenizador Punkt
       en español) — nunca se corta a mitad de una oración.
    2. Las oraciones se acumulan en un fragmento hasta acercarse a
       CHUNK_SIZE caracteres (definido en config.py); ahí se cierra el
       fragmento y se abre el siguiente.
    3. Cada fragmento nuevo arranca repitiendo las últimas oraciones del
       fragmento anterior, hasta sumar ~CHUNK_OVERLAP caracteres, para no
       perder contexto que haya quedado justo en el borde del corte.
    4. Fragmentos finales demasiado cortos (sobrantes al final de un
       documento) se fusionan con el fragmento previo en vez de quedar
       como un fragmento casi vacío.

  Cada fragmento guarda de qué documento vino (doc_id, título, autor) y su
  posición dentro del documento — esa metadata es la que permite citar la
  fuente exacta en la Fase 6.

Uso:
    python src/chunk.py
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize

from config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
LONGITUD_MINIMA_FRAGMENTO = 200  # caracteres; fragmentos más chicos se fusionan con el anterior


def asegurar_recursos_nltk() -> None:
    """
    Descarga el tokenizador de oraciones (Punkt, español) si no está disponible
    localmente. Se verifica el idioma específico ("spanish"), no solo la carpeta
    genérica "punkt_tab": esa carpeta puede existir de forma parcial (por ejemplo,
    con datos de otro idioma de una instalación previa) y dar un falso positivo
    que se salta la descarga real que hace falta.
    """
    recurso_especifico = "tokenizers/punkt_tab/spanish/"
    try:
        nltk.data.find(recurso_especifico)
        return
    except LookupError:
        pass

    logger.info("Descargando el tokenizador de oraciones de NLTK (una sola vez)...")
    nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find(recurso_especifico)
    except LookupError as e:
        logger.error(
            "No se pudo obtener el tokenizador de NLTK en español tras intentar "
            "descargarlo. Prueba manualmente en una terminal:\n"
            "    python -c \"import nltk; nltk.download('punkt_tab')\"\n"
            "Si tu red/firewall bloquea la descarga, revisa la conexión o usa una VPN."
        )
        raise SystemExit(1) from e


@dataclass
class Fragmento:
    chunk_id: str
    doc_id: str
    titulo: str
    autor: str
    posicion: int
    texto: str
    num_caracteres: int


def dividir_en_oraciones(texto: str) -> list[str]:
    # El texto trae un salto de línea por cada línea física del PDF original
    # (ver Fase 2), no por párrafo. Se unen en espacios antes de tokenizar
    # por oración, porque una oración real puede haber quedado partida en
    # varias líneas del layout original.
    texto_unido = " ".join(linea.strip() for linea in texto.split("\n") if linea.strip())
    return sent_tokenize(texto_unido, language="spanish")


def armar_fragmentos(oraciones: list[str]) -> list[str]:
    """Empaqueta oraciones completas en fragmentos de ~CHUNK_SIZE caracteres con overlap."""
    fragmentos: list[str] = []
    actual: list[str] = []
    largo_actual = 0

    for oracion in oraciones:
        if largo_actual + len(oracion) > CHUNK_SIZE and actual:
            fragmentos.append(" ".join(actual))

            # Overlap: se retienen oraciones del final del fragmento recién
            # cerrado hasta juntar ~CHUNK_OVERLAP caracteres, y se usan como
            # punto de partida del fragmento siguiente.
            colas: list[str] = []
            largo_cola = 0
            for o in reversed(actual):
                if largo_cola >= CHUNK_OVERLAP:
                    break
                colas.insert(0, o)
                largo_cola += len(o)

            actual = colas
            largo_actual = largo_cola

        actual.append(oracion)
        largo_actual += len(oracion)

    if actual:
        fragmentos.append(" ".join(actual))

    # Fusiona fragmentos finales demasiado cortos con el fragmento anterior.
    fragmentos_finales: list[str] = []
    for frag in fragmentos:
        if fragmentos_finales and len(frag) < LONGITUD_MINIMA_FRAGMENTO:
            fragmentos_finales[-1] = fragmentos_finales[-1] + " " + frag
        else:
            fragmentos_finales.append(frag)

    return fragmentos_finales


def procesar_documento(ruta_json: Path) -> list[Fragmento]:
    doc = json.loads(ruta_json.read_text(encoding="utf-8"))
    oraciones = dividir_en_oraciones(doc["texto"])

    if not oraciones:
        logger.warning("%s no produjo oraciones — se omite.", ruta_json.name)
        return []

    textos_fragmentos = armar_fragmentos(oraciones)

    return [
        Fragmento(
            chunk_id=f"{doc['doc_id']}__{i:03d}",
            doc_id=doc["doc_id"],
            titulo=doc["titulo"],
            autor=doc["autor"],
            posicion=i,
            texto=texto,
            num_caracteres=len(texto),
        )
        for i, texto in enumerate(textos_fragmentos)
    ]


def main():
    asegurar_recursos_nltk()

    archivos_json = sorted(DATA_PROCESSED_DIR.glob("*.json"))
    if not archivos_json:
        logger.warning("No hay documentos procesados en %s. Corre primero src/ingest.py.", DATA_PROCESSED_DIR)
        return

    todos_los_fragmentos: list[Fragmento] = []
    for ruta in archivos_json:
        fragmentos = procesar_documento(ruta)
        todos_los_fragmentos.extend(fragmentos)
        logger.info("%s -> %d fragmentos", ruta.name, len(fragmentos))

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for frag in todos_los_fragmentos:
            f.write(json.dumps(asdict(frag), ensure_ascii=False) + "\n")

    tamanos = [f.num_caracteres for f in todos_los_fragmentos]
    logger.info(
        "Listo. %d documentos -> %d fragmentos en total. Tamaño promedio: %.0f caracteres (min %d, max %d).",
        len(archivos_json), len(todos_los_fragmentos),
        sum(tamanos) / len(tamanos) if tamanos else 0,
        min(tamanos, default=0), max(tamanos, default=0),
    )
    logger.info("Guardado en: %s", CHUNKS_PATH)


if __name__ == "__main__":
    main()

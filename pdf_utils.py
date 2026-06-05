import os
import re
import tempfile

import fitz  # pymupdf
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


def extract_epub_pages(epub_bytes: bytes) -> list[dict]:
    """Extrai capítulos de um EPUB em ordem de leitura."""
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
        tmp.write(epub_bytes)
        tmp_path = tmp.name
    try:
        book = epub.read_epub(tmp_path)
        pages = []
        for item_id, _ in book.spine:
            item = book.get_item_with_id(item_id)
            if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            try:
                content = item.get_content()
            except Exception:
                continue
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup.find_all(["nav", "aside", "script", "style"]):
                tag.decompose()
            # Títulos: concatena filhos diretos preservando espaços do texto original
            # (resolve letras decorativas separadas como "B estiário" → "Bestiário")
            for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
                raw = "".join(c.get_text() if hasattr(c, "get_text") else str(c)
                              for c in heading.children)
                heading_text = re.sub(r"\s+", " ", raw).strip()
                heading.replace_with(" " + heading_text + ". ")
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            # Fix drop caps no corpo: "H á" → "Há", "C omeçou" → "Começou"
            text = re.sub(r"\b([BCDFGHJKLMNPQRSTVWXYZ]) ([a-záéíóúàãõâêô])", r"\1\2", text)
            if len(text.split()) > 20:
                pages.append({"index": len(pages), "text": text, "words": len(text.split())})
        return pages
    finally:
        os.unlink(tmp_path)


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = _extract_page_text(page)
        if text:
            pages.append({"index": i, "text": text, "words": len(text.split())})
    doc.close()
    return pages


def _extract_page_text(page) -> str:
    page_h = page.rect.height

    # get_text("blocks") retorna (x0, y0, x1, y1, texto, nº_bloco, tipo)
    # tipo 0 = texto, tipo 1 = imagem
    blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))

    paragraphs = []
    for x0, y0, x1, y1, text, _bno, btype in blocks:
        if btype != 0:
            continue

        text = text.strip()
        if not text:
            continue

        # Remove números de página e cabeçalhos/rodapés:
        # blocos curtos nas margens superior (7%) ou inferior (7%)
        in_top    = y1 < page_h * 0.07
        in_bottom = y0 > page_h * 0.93
        if (in_top or in_bottom) and len(text) < 80:
            continue

        # Remove linhas que são só números (número de página no meio)
        if re.match(r'^\d+$', text.strip()):
            continue

        # Une hifenações de fim de linha ("pala-\nvra" → "palavra")
        text = re.sub(r'-\n', '', text)
        # Colapsa quebras de linha restantes em espaço
        text = re.sub(r'\n+', ' ', text)
        # Normaliza espaços
        text = re.sub(r' {2,}', ' ', text).strip()

        if text:
            paragraphs.append(text)

    # Parágrafos separados por ponto-e-espaço para o split_sentences() funcionar
    return '  '.join(paragraphs)

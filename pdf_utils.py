import re
import fitz  # pymupdf


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

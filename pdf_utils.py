import re
import fitz  # pymupdf


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        raw = page.get_text()
        text = clean_pdf_text(raw)
        if text:
            pages.append({"index": i, "text": text, "words": len(text.split())})
    doc.close()
    return pages


def clean_pdf_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)       # une hifenações de fim de linha
    text = re.sub(r"\n+", " ", text)      # quebras de linha → espaço
    text = re.sub(r" {2,}", " ", text)    # espaços múltiplos → um
    return text.strip()

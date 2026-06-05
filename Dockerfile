FROM python:3.11-slim

WORKDIR /app

COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY server.py tts_engine.py pdf_utils.py index.html ./

ENV BOOKS_DIR=/data/books

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]

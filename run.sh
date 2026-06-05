#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

VENV_DIR=".venv"
PYTHON="$(PYENV_VERSION=3.11.9 pyenv which python 2>/dev/null || python3)"

# Cria o virtualenv na primeira execução
if [ ! -d "$VENV_DIR" ]; then
  echo "🔧 Criando ambiente virtual com Python $($PYTHON --version)..."
  "$PYTHON" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "📦 Instalando dependências..."
pip install -r requirements.txt -q

echo ""
echo "🎵 TTS Player iniciando em http://localhost:8000"
echo "   (Na primeira execução, os modelos Kokoro (~300MB) serão baixados automaticamente)"
echo "   Pressione Ctrl+C para parar."
echo ""

uvicorn server:app --host 0.0.0.0 --port 8000 --reload

"""
Configuração centralizada do projeto.
Todas as constantes (URLs, modelos, dimensões) ficam aqui.
"""

import os

# ── PostgreSQL ───────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/cgr_assistant",
)

# ── Ollama ───────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "llama3.2")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "768"))

# ── Diretórios ───────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
UPLOAD_DIR: str = os.path.join(DATA_DIR, "uploads")
CHUNKS_DIR: str = os.path.join(DATA_DIR, "chunks")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

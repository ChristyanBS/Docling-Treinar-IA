"""
Serviço de embeddings via Ollama (nomic-embed-text).
"""
import logging
from typing import List, Optional

import httpx

from app.config import OLLAMA_BASE_URL, EMBED_MODEL

logger = logging.getLogger(__name__)

_TIMEOUT = 60.0


def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """Gera embeddings em batch via Ollama /api/embed."""
    if not texts:
        return []
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": texts},
            )
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if embeddings and len(embeddings) == len(texts):
                    return embeddings

            # Fallback: um por um via /api/embeddings
            logger.warning("Batch embed falhou, tentando individual...")
            results = []
            for t in texts:
                r = client.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": t},
                )
                if r.status_code == 200:
                    emb = r.json().get("embedding", [])
                    results.append(emb)
                else:
                    results.append(None)
            if all(r is not None for r in results):
                return results

    except Exception as e:
        logger.error(f"Erro ao gerar embeddings: {e}")

    return None


def embed_single(text: str) -> Optional[List[float]]:
    """Gera embedding para um texto único."""
    result = embed_texts([text])
    if result and len(result) == 1:
        return result[0]
    return None

"""
Serviço de geração de embeddings via Ollama.
Usa o endpoint /api/embed (Ollama ≥ 0.4) com fallback para /api/embeddings.
"""

import httpx
import logging
from typing import List, Optional

from app.config import OLLAMA_BASE_URL, EMBED_MODEL, EMBED_DIM

logger = logging.getLogger(__name__)

_TIMEOUT = 120.0  # embeddings podem demorar na primeira vez (download do modelo)


def _zero_vector() -> List[float]:
    """Vetor zero usado como fallback quando a geração falha."""
    return [0.0] * EMBED_DIM


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Gera embeddings para uma lista de textos via Ollama.
    Retorna uma lista de vetores (mesma ordem de `texts`).
    Em caso de falha, retorna vetores zero para cada texto.
    """
    if not texts:
        return []

    # Limpar textos vazios
    cleaned = [t.strip() if t and t.strip() else "." for t in texts]

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            # Tentar endpoint /api/embed (Ollama ≥ 0.4, suporta batch)
            resp = client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": cleaned},
            )
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if embeddings and len(embeddings) == len(texts):
                    return embeddings

            # Fallback: /api/embeddings (versão mais antiga, um por vez)
            logger.warning("Fallback para /api/embeddings (um texto por vez)")
            results: List[List[float]] = []
            for text in cleaned:
                resp2 = client.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": text},
                )
                if resp2.status_code == 200:
                    emb = resp2.json().get("embedding", [])
                    results.append(emb if emb else _zero_vector())
                else:
                    results.append(_zero_vector())
            return results

    except Exception as e:
        logger.error("Erro ao gerar embeddings: %s", e)
        return [_zero_vector() for _ in texts]


def embed_single(text: str) -> List[float]:
    """Gera embedding para um único texto."""
    results = embed_texts([text])
    return results[0] if results else _zero_vector()

"""
Serviço de comunicação com o Ollama (chat streaming).
"""
import json
import logging
from typing import Generator

import httpx

from app.config import OLLAMA_BASE_URL, CHAT_MODEL

logger = logging.getLogger(__name__)


def check_ollama() -> dict:
    """Verifica se o Ollama está online."""
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return {"online": True, "models": models}
    except Exception:
        pass
    return {"online": False, "models": []}


def chat_stream(messages: list, model: str = None) -> Generator[str, None, None]:
    """Envia mensagens para o Ollama e retorna resposta em streaming."""
    model = model or CHAT_MODEL
    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": model, "messages": messages, "stream": True},
            ) as response:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                yield token
                            if data.get("done", False):
                                return
                        except json.JSONDecodeError:
                            continue
    except httpx.ConnectError:
        yield "❌ Erro: Ollama não está rodando. Execute `ollama serve` no terminal."
    except Exception as e:
        yield f"❌ Erro: {str(e)}"
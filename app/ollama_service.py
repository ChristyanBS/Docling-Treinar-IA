"""
Serviço de comunicação com o Ollama — inclui streaming token-a-token.
"""

import httpx
import json
from typing import Optional, List, AsyncGenerator

OLLAMA_BASE_URL = "http://localhost:11434"


async def check_ollama_status() -> dict:
    """Verifica se o Ollama está online e retorna modelos."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return {
                    "status": "connected",
                    "models": [m["name"] for m in models]
                }
    except Exception as e:
        return {"status": "offline", "models": [], "error": str(e)}
    return {"status": "offline", "models": []}


async def list_models() -> List[str]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                return [m["name"] for m in response.json().get("models", [])]
    except Exception:
        pass
    return []


async def chat_stream(
    messages: List[dict],
    model: str = "llama3.2",
) -> AsyncGenerator[str, None]:
    """Stream tokens from Ollama chat API."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": model, "messages": messages, "stream": True},
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    except httpx.ConnectError:
        yield "[Erro: Ollama não está rodando. Execute 'ollama serve']"
    except httpx.TimeoutException:
        yield "[Erro: Timeout — modelo pode estar carregando]"
    except Exception as e:
        yield f"[Erro: {str(e)}]"


async def chat_with_ollama(
    prompt: str,
    model: str = "llama3.2",
    context: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Envia mensagem e retorna resposta completa (sem stream)."""
    messages = []
    sys_content = system_prompt or "Você é um assistente inteligente. Responda em português brasileiro."
    if context:
        sys_content += f"\n\nContexto dos documentos:\n\n{context}"
    messages.append({"role": "system", "content": sys_content})
    messages.append({"role": "user", "content": prompt})

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
            )
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "Sem resposta.")
            return f"Erro Ollama: HTTP {response.status_code}"
    except httpx.ConnectError:
        return "Erro: Ollama offline. Execute 'ollama serve'."
    except httpx.TimeoutException:
        return "Erro: Timeout — tente novamente."
    except Exception as e:
        return f"Erro: {str(e)}"
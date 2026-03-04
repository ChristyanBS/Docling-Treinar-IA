"""
Serviço de comunicação com o Ollama — inclui streaming token-a-token.
"""

import httpx
import json
from typing import Optional, List, AsyncGenerator
from app.config import OLLAMA_BASE_URL, CHAT_MODEL
from app.database import SessionLocal, ChatMessage


def get_history_messages(session_id: str, limit: int = 10) -> List[dict]:
    """Busca as últimas `limit` mensagens de uma sessão de chat."""
    db = SessionLocal()
    try:
        # Busca as últimas mensagens daquela sessão específica
        history = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at.desc()).limit(limit).all()
        
        # Converte para o formato que o Ollama aceita e inverte para manter a ordem cronológica
        return [{"role": m.role, "content": m.content} for m in reversed(history)]
    finally:
        db.close()

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
    session_id: str,
    model: str = CHAT_MODEL,
) -> AsyncGenerator[str, None]:
    """Stream tokens from Ollama chat API, injetando histórico da sessão."""
    # Injeta o histórico da conversa antes da última mensagem do usuário
    final_messages = messages
    if session_id and len(messages) > 0:
        # A última mensagem é a do usuário atual
        user_message = final_messages.pop()
        # O histórico não inclui a mensagem atual (será salva depois)
        history = get_history_messages(session_id, limit=10)
        final_messages.extend(history)
        final_messages.append(user_message)

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": model, "messages": final_messages, "stream": True},
            ) as resp:
                resp.raise_for_status()
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
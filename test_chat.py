"""Test the chat SSE endpoint and health check."""
import httpx
import sys

BASE = "http://localhost:8000"

# Health check
try:
    r = httpx.get(f"{BASE}/api/health", timeout=5)
    print("HEALTH:", r.json())
except Exception as e:
    print(f"Server not reachable: {e}")
    sys.exit(1)

# Ollama status
try:
    r = httpx.get(f"{BASE}/api/ollama/status", timeout=5)
    print("OLLAMA:", r.json())
except Exception as e:
    print(f"Ollama check failed: {e}")

# Knowledge stats
try:
    r = httpx.get(f"{BASE}/api/knowledge/stats", timeout=5)
    print("KNOWLEDGE:", r.json())
except Exception as e:
    print(f"Knowledge check failed: {e}")

# Test chat with SSE streaming
print("\n=== TESTING CHAT SSE ===")
try:
    with httpx.stream(
        "POST",
        f"{BASE}/api/chat",
        json={"message": "Oi, tudo bem?"},
        timeout=30,
    ) as resp:
        print(f"Status: {resp.status_code}")
        print(f"Headers: {dict(resp.headers)}")
        for line in resp.iter_lines():
            if line.strip():
                print(f"  SSE> {line}")
    print("=== CHAT OK ===")
except Exception as e:
    print(f"CHAT ERROR: {e}")

# Test sessions
try:
    r = httpx.get(f"{BASE}/api/chat/sessions", timeout=5)
    print("SESSIONS:", r.json())
except Exception as e:
    print(f"Sessions check failed: {e}")

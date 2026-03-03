import httpx
try:
    r = httpx.get("http://localhost:8000/api/health", timeout=5)
    print("HEALTH:", r.json())
    r2 = httpx.get("http://localhost:8000/api/ollama/status", timeout=5)
    print("OLLAMA:", r2.json())
    r3 = httpx.get("http://localhost:8000/api/knowledge/stats", timeout=5)
    print("KNOWLEDGE:", r3.json())
    print("\n=== SERVER IS RUNNING OK ===")
except Exception as e:
    print("ERROR:", e)

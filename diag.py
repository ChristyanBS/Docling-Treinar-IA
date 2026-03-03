import httpx, traceback
out = []
BASE = "http://localhost:8000"

# 1) Health
try:
    r = httpx.get(f"{BASE}/api/health", timeout=5)
    out.append(f"HEALTH: {r.status_code} {r.json()}")
except Exception as e:
    out.append(f"HEALTH FAIL: {e}")

# 2) Ollama status
try:
    r = httpx.get(f"{BASE}/api/ollama/status", timeout=10)
    out.append(f"OLLAMA: {r.status_code} {r.json()}")
except Exception as e:
    out.append(f"OLLAMA FAIL: {e}")

# 3) Ollama direct check
try:
    r = httpx.get("http://localhost:11434/api/tags", timeout=5)
    out.append(f"OLLAMA DIRECT: {r.status_code} models={[m['name'] for m in r.json().get('models',[])]}")
except Exception as e:
    out.append(f"OLLAMA DIRECT FAIL: {e}")

# 4) Knowledge stats
try:
    r = httpx.get(f"{BASE}/api/knowledge/stats", timeout=5)
    out.append(f"KNOWLEDGE: {r.status_code} {r.json()}")
except Exception as e:
    out.append(f"KNOWLEDGE FAIL: {e}")

# 5) Quick chat test
try:
    with httpx.stream("POST", f"{BASE}/api/chat", json={"message": "oi"}, timeout=30) as resp:
        out.append(f"CHAT: status={resp.status_code}")
        lines = []
        for line in resp.iter_lines():
            if line.strip():
                lines.append(line)
                if len(lines) >= 10:
                    break
        for l in lines:
            out.append(f"  {l}")
except Exception as e:
    out.append(f"CHAT FAIL: {e}\n{traceback.format_exc()}")

with open("diag_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done - see diag_result.txt")

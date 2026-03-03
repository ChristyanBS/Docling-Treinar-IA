"""Quick chat test - saves output to test_result.txt"""
import httpx

out = []
BASE = "http://localhost:8000"

try:
    r = httpx.get(f"{BASE}/api/health", timeout=5)
    out.append(f"HEALTH: {r.json()}")
except Exception as e:
    out.append(f"HEALTH FAIL: {e}")

try:
    with httpx.stream("POST", f"{BASE}/api/chat", json={"message": "Oi"}, timeout=60) as resp:
        out.append(f"CHAT Status: {resp.status_code}")
        for line in resp.iter_lines():
            if line.strip():
                out.append(f"  {line}")
        out.append("CHAT OK")
except Exception as e:
    out.append(f"CHAT ERROR: {e}")

try:
    r = httpx.get(f"{BASE}/api/chat/sessions", timeout=5)
    out.append(f"SESSIONS: {r.json()}")
except Exception as e:
    out.append(f"SESSIONS FAIL: {e}")

with open("test_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))

print("Done - see test_result.txt")

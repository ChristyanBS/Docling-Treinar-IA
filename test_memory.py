"""Test memory features: chat remembers, learns from conversation, recalls past."""
import httpx

out = []
BASE = "http://localhost:8000"

# 1) Health
try:
    r = httpx.get(f"{BASE}/api/health", timeout=5)
    out.append(f"1. HEALTH: {r.json()}")
except Exception as e:
    out.append(f"1. HEALTH FAIL: {e}")

# 2) Chat 1: Tell the AI something to remember
out.append("\n=== CHAT 1: Telling AI my name ===")
session1 = None
try:
    with httpx.stream("POST", f"{BASE}/api/chat",
                       json={"message": "Meu nome é Carlos e eu trabalho na CGR Telecom como engenheiro de redes. Lembre disso."},
                       timeout=60) as resp:
        tokens = []
        for line in resp.iter_lines():
            if line.strip():
                if line.startswith("event: done"):
                    pass
                elif line.startswith("data: "):
                    import json
                    try:
                        d = json.loads(line[6:])
                        if "token" in d:
                            tokens.append(d["token"])
                        if "session_id" in d:
                            session1 = d["session_id"]
                    except:
                        pass
        response1 = "".join(tokens)
        out.append(f"   AI response: {response1[:200]}")
        out.append(f"   Session: {session1}")
except Exception as e:
    out.append(f"   ERROR: {e}")

# 3) Check memories were saved
out.append("\n=== MEMORIES AFTER CHAT 1 ===")
try:
    r = httpx.get(f"{BASE}/api/memories", timeout=5)
    mems = r.json()
    out.append(f"   Total memories: {mems.get('total', 0)}")
    for m in mems.get("memories", []):
        out.append(f"   - [{m['category']}] {m['fact']}")
except Exception as e:
    out.append(f"   ERROR: {e}")

# 4) Chat 2: NEW session - ask if the AI remembers
out.append("\n=== CHAT 2: New session asking 'qual meu nome?' ===")
try:
    with httpx.stream("POST", f"{BASE}/api/chat",
                       json={"message": "Qual é o meu nome e onde eu trabalho?"},
                       timeout=60) as resp:
        tokens = []
        sources = []
        for line in resp.iter_lines():
            if line.strip():
                if line.startswith("data: "):
                    try:
                        d = json.loads(line[6:])
                        if "token" in d:
                            tokens.append(d["token"])
                        if "sources" in d:
                            sources = d["sources"]
                    except:
                        pass
        response2 = "".join(tokens)
        out.append(f"   AI response: {response2[:300]}")
        out.append(f"   Sources: {sources}")
except Exception as e:
    out.append(f"   ERROR: {e}")

# 5) Knowledge stats
out.append("\n=== KNOWLEDGE STATS ===")
try:
    r = httpx.get(f"{BASE}/api/knowledge/stats", timeout=5)
    out.append(f"   {r.json()}")
except Exception as e:
    out.append(f"   ERROR: {e}")

with open("test_memory_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done - see test_memory_result.txt")

"""Quick test: ingest JSONL into SQLite + Ollama embeddings."""
import os
os.environ["PYTHONUTF8"] = "1"

from pathlib import Path
from app.vectorstore import ingest_jsonl, get_stats, query_knowledge

jsonl = Path("./saida_ia/conhecimento_cgr.jsonl")
print("=== Ingest Test ===")
count = ingest_jsonl(jsonl, on_log=print)
print(f"Inserted: {count}")
print(f"Stats: {get_stats()}")

print("\n=== Query Test ===")
results = query_knowledge("O que é OSPF?", n_results=3)
for r in results:
    print(f"  [{r['source']}] dist={r['distance']:.3f} => {r['text'][:80]}...")
print("\nDone!")

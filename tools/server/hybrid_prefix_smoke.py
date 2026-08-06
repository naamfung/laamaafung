# hybrid prefix-sharing smoke test for llama-server (qwen35 hybrid model)
# 1. request 1: long prefix on slot 0 (prefill writes recurrent chunk snapshots)
# 2. request 2: same prefix on slot 1 (new seq) -> prefix sharing should hit
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8090"

def post(path, payload, timeout=900):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())

def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode())

def main():
    for _ in range(120):
        try:
            props = get("/props")
            print("server ready, model:", props.get("model_path"))
            break
        except Exception:
            time.sleep(2)
    else:
        print("server did not become ready")
        sys.exit(1)

    # ~600+ tokens so the first ubatch (512 tokens) lands on a chunk boundary
    para = ("The hybrid model combines linear attention with standard attention. "
            "Recurrent layers maintain a compact state instead of a growing cache. "
            "Prefix sharing skips recomputation of already-seen chunks. ")
    prefix = (para * 20)[:3000]
    suffix_a = " Now answer with the single word: alpha."
    suffix_b = " Now answer with the single word: bravo."

    r1 = post("/completion", {
        "prompt": prefix + suffix_a,
        "n_predict": 8,
        "temperature": 0,
        "cache_prompt": True,
        "id_slot": 0,
    })
    print(f"[req1 slot0] -> {r1.get('content','')[:80]!r}")

    r2 = post("/completion", {
        "prompt": prefix + suffix_b,
        "n_predict": 8,
        "temperature": 0,
        "cache_prompt": True,
        "id_slot": 1,
    })
    print(f"[req2 slot1] -> {r2.get('content','')[:80]!r}")

    with open("hybrid_smoke_out.json", "w", encoding="utf-8") as f:
        json.dump({"req1": r1.get("content", ""), "req2": r2.get("content", "")}, f, ensure_ascii=False, indent=2)
    print("done")

if __name__ == "__main__":
    main()

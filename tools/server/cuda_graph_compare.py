# CUDA Graph on/off comparison for the paged KV cache (hybrid qwen35).
# usage: python cuda_graph_compare.py <port>
# requests: slot0 prefill prefix; slot1 same prefix + suffix (prefix sharing)
import json
import sys
import time
import urllib.request

BASE = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else 8090}"

def post(p, timeout=600):
    req = urllib.request.Request(
        BASE + "/completion",
        data=json.dumps(p).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())

def main():
    for _ in range(120):
        try:
            get("/props")
            break
        except Exception:
            time.sleep(2)
    else:
        print("server not ready")
        sys.exit(1)

    para = ("The hybrid model combines linear attention with standard attention. "
            "Recurrent layers maintain a compact state instead of a growing cache. ")
    prefix = (para * 12)[:2400]  # ~350 tokens, single ubatch

    t0 = time.time()
    r1 = post({"prompt": prefix + " Answer: alpha.", "n_predict": 4, "temperature": 0, "id_slot": 0})
    dt1 = time.time() - t0

    t0 = time.time()
    r2 = post({"prompt": prefix + " Answer: bravo.", "n_predict": 4, "temperature": 0, "id_slot": 1})
    dt2 = time.time() - t0

    print(f"prefill={dt1:.3f}s shared_decode={dt2:.3f}s")
    print("out1:", repr(r1.get("content", "")[:40]))
    print("out2:", repr(r2.get("content", "")[:40]))
    with open("cg_out.json", "w", encoding="utf-8") as f:
        json.dump({"out1": r1.get("content", ""), "out2": r2.get("content", "")}, f, ensure_ascii=False)

if __name__ == "__main__":
    main()

# minimal HTTP client for llama-server smoke tests (stdlib only)
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8090"

def post(path, payload, timeout=600):
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
    # wait for server
    for _ in range(60):
        try:
            props = get("/props")
            print("server ready, model:", props.get("model_path"))
            break
        except Exception:
            time.sleep(2)
    else:
        print("server did not become ready")
        sys.exit(1)

    # simple generation on slot 0
    t0 = time.time()
    r = post("/completion", {
        "prompt": "Hello, tell me about yourself in one sentence.",
        "n_predict": 24,
        "temperature": 0,
        "cache_prompt": True,
    })
    dt = time.time() - t0
    text = r.get("content", "")
    print(f"[slot0] {dt:.1f}s, {len(text)} chars")
    print("  text:", text[:120].replace("\n", " "))

    # second request: same prompt, prefix cache should hit
    t0 = time.time()
    r2 = post("/completion", {
        "prompt": "Hello, tell me about yourself in one sentence.",
        "n_predict": 24,
        "temperature": 0,
        "cache_prompt": True,
    })
    dt2 = time.time() - t0
    text2 = r2.get("content", "")
    print(f"[slot0-again] {dt2:.1f}s (cached), {len(text2)} chars")
    print("  text:", text2[:120].replace("\n", " "))
    print("  prefix hit:", r2.get("timings", {}).get("prompt_n", -1))

    # parallel slot: slot 1
    t0 = time.time()
    r3 = post("/completion", {
        "prompt": "What is 2+2? Answer briefly.",
        "n_predict": 16,
        "temperature": 0,
        "cache_prompt": True,
    })
    dt3 = time.time() - t0
    print(f"[slot1] {dt3:.1f}s")
    print("  text:", r3.get("content", "")[:120].replace("\n", " "))

    print("SMOKE OK")

if __name__ == "__main__":
    main()

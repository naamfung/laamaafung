# CUDA Graph shape-change recapture benchmark for the paged KV cache.
#
# usage: python shape_bench.py <port> <label>
#   label: e.g. "graph-on" / "graph-off". results appended to shape_bench.csv
#          (or printed if --csv path given as third arg)
#
# measures prompt-eval / decode latency under two request patterns:
#   * mixed:  concurrency cycles 1..4, prompt lengths cycle {256,512,1024,2048}
#             -> ubatch shape changes constantly -> forces graph recaptures
#   * fixed:  constant 2 concurrent slots, constant prompt length -> graph reuse
# the difference between the two (and between graph-on/off runs) bounds the
# recapture cost.
import csv
import json
import sys
import time
import urllib.request

BASE = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else 8090}"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "graph-on"
CSV = sys.argv[3] if len(sys.argv) > 3 else "shape_bench.csv"

PARA = ("The hybrid model combines linear attention with standard attention. "
        "Recurrent layers maintain a compact state instead of a growing cache. ")

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

def prompt_of(n_tokens, seed):
    # seed makes every prompt unique so prefix sharing never kicks in and the
    # measured prefill reflects a real (unshared) shape change
    s = (PARA + f" Token seed {seed}. ") * ((n_tokens + 60) // 140)
    return s[: n_tokens * 4]  # ~4 chars/token

def one(seq, n_tok, n_pred):
    r = post({"prompt": prompt_of(n_tok, hash((LABEL, seq, n_tok, time.time())) & 0xffffff),
              "n_predict": n_pred, "temperature": 0, "id_slot": seq})
    t = r.get("timings", {})
    return t.get("prompt_ms", 0), t.get("predicted_ms", 0), t.get("predicted_n", 1), t.get("cache_n", 0)

def run(rows):
    # mixed shape pattern: 20 rounds, concurrency cycles 1..4
    for round_i in range(20):
        k = (round_i % 4) + 1
        for i in range(k):
            n_tok = [256, 512, 1024, 2048][(round_i + i) % 4]
            pe, pd, pn, cache_n = one(i, n_tok, 16)
            rows.append(("mixed", k, n_tok, pe, pd / max(pn, 1)))

    # fixed shape pattern: 20 rounds, 2 concurrent, constant 512
    for _ in range(20):
        for i in range(2):
            pe, pd, pn, cache_n = one(i, 512, 16)
            rows.append(("fixed", 2, 512, pe, pd / max(pn, 1)))

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

    rows = []
    run(rows)

    # aggregate per pattern
    agg = {}
    for pat, k, n, pe, pd in rows:
        key = (pat, k, n)
        if key not in agg:
            agg[key] = [0, 0, 0]  # pe sum, pd sum, count
        agg[key][0] += pe
        agg[key][1] += pd
        agg[key][2] += 1

    print(f"== {LABEL}: {len(rows)} requests ==")
    for pat in ("mixed", "fixed"):
        vals = [(k, n, a[0] / a[2], a[1] / a[2], a[2]) for (p, k, n), a in agg.items() if p == pat]
        if pat == "mixed":
            vals.sort()
        else:
            vals.sort()
        for k, n, pe, pd, cnt in vals:
            print(f"{pat:5s} concurrency={k} prompt_len={n:5d} prompt_eval={pe:8.1f}ms decode={pd:7.2f}ms/tok n={cnt}")

    with open(CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["label", "pattern", "concurrency", "prompt_len", "prompt_eval_ms", "decode_ms_per_tok", "n"])
        for pat in ("mixed", "fixed"):
            for (p, k, n), a in sorted(agg.items()):
                if p == pat:
                    w.writerow([LABEL, pat, k, n, round(a[0] / a[2], 1), round(a[1] / a[2], 2), a[2]])

if __name__ == "__main__":
    main()

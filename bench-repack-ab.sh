#!/bin/bash
# A/B test: repack vs no-repack, same threads=10, same load-mode none.
BENCH="d:/Programs/llama-cpp-repos/laamaafung/build-v12/bin/Release/llama-bench.exe"
MODEL="D:/models/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf"

echo "=== A: NO repack (--no-extra-bufts), threads=10 ==="
time "$BENCH" -m "$MODEL" -ngl 99 -fa 1 -b 1024 -ub 1024 -r 3 \
  -ctk f16 -ctv f16 -p 512 -n 128 \
  --load-mode none --n-cpu-moe 33 --threads 10 --no-extra-bufts 2>&1

echo ""
echo "=== B: WITH repack (default), threads=10 ==="
time "$BENCH" -m "$MODEL" -ngl 99 -fa 1 -b 1024 -ub 1024 -r 3 \
  -ctk f16 -ctv f16 -p 512 -n 128 \
  --load-mode none --n-cpu-moe 33 --threads 10 2>&1

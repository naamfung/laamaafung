#!/bin/bash
# Test repack optimization: compare with previous baseline (28.12 t/s for tg128 f16).
# Using --load-mode none to avoid mmap+repack memory doubling.
BENCH="d:/Programs/llama-cpp-repos/laamaafung/build-v12/bin/Release/llama-bench.exe"
MODEL="D:/models/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf"

echo "=== 35B MoE with repack enabled (load-mode none) ==="
echo "--- f16 KV, threads=18, ncmoe=33 ---"
"$BENCH" -m "$MODEL" -ngl 99 -fa 1 -b 1024 -ub 1024 -r 3 \
  -ctk f16 -ctv f16 -p 512 -n 128 \
  --load-mode none --n-cpu-moe 33 --threads 18 2>&1

echo ""
echo "--- f16 KV, threads=10, ncmoe=33 ---"
"$BENCH" -m "$MODEL" -ngl 99 -fa 1 -b 1024 -ub 1024 -r 3 \
  -ctk f16 -ctv f16 -p 512 -n 128 \
  --load-mode none --n-cpu-moe 33 --threads 10 2>&1

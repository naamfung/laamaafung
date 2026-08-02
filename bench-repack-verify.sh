#!/bin/bash
# Verify repack: if enabled, mlock-ram + repack buffer should cause memory pressure.
# Also test with -v to see if repack log appears.
BENCH="d:/Programs/llama-cpp-repos/laamaafung/build-v12/bin/Release/llama-bench.exe"
MODEL="D:/models/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf"

echo "=== Test 1: mlock-ram + threads=10 (OOM means repack active) ==="
time "$BENCH" -m "$MODEL" -ngl 99 -fa 1 -b 1024 -ub 1024 -r 3 \
  -ctk f16 -ctv f16 -p 512 -n 128 \
  --load-mode mlock-ram --n-cpu-moe 33 --threads 10 -v 2>&1 | head -60

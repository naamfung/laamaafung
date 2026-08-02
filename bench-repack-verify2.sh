#!/bin/bash
# Verify repack with verbose output, save to file for inspection.
BENCH="d:/Programs/llama-cpp-repos/laamaafung/build-v12/bin/Release/llama-bench.exe"
MODEL="D:/models/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf"
OUT="d:/Programs/llama-cpp-repos/laamaafung/repack-test.log"

"$BENCH" -m "$MODEL" -ngl 99 -fa 1 -b 1024 -ub 1024 -r 2 \
  -ctk f16 -ctv f16 -p 512 -n 128 \
  --load-mode none --n-cpu-moe 33 --threads 10 -v \
  > "$OUT" 2>&1

echo "=== Done. Exit: $? ==="
echo "=== repack lines: ==="
grep -i "repack" "$OUT" | head -20
echo "=== t/s lines: ==="
grep -i "t/s" "$OUT" | head -5
echo "=== error lines: ==="
grep -i "error\|OOM\|failed" "$OUT" | head -10

#!/bin/bash
# 9B 生产基准矩阵：按 config.yaml「Qwen3.5-9B-Uncensored-Aggressive」方案复刻（72.38 t/s 锚点）+ 增量
PY=/c/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe
BIN26=G:/Agents/kvmem-works/laamaafung/build-v26/bin/llama-server.exe
BIN25=G:/Agents/kvmem-works/wt-v25/build-v25/bin/llama-server.exe
M9="K:/models9/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
MM9="C:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
TMPL_AG="C:/iStartModel/tmpl/Qwen-Agentic-HONT.jinja"
TMP=/g/Agents/kvmem-works/_tmp
TMPW=G:/Agents/kvmem-works/_tmp
LOGF="$1"

# 9B 生产基准（config 复刻）
BASE=(-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified \
      -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram \
      --cache-prompt --cache-ram 8192 --fit on -c 131072)

run_test() {
  local BIN="$1"; shift
  local tag="$1"; shift
  local -a args=("$@")
  echo "=== [$tag] start $(date +%H:%M:%S) ===" | tee -a "$LOGF"
  "$BIN" -m "$M9" --chat-template-file "$TMPL_AG" "${args[@]}" --port 9670 > "$TMP/mx3-$tag.log" 2>&1 &
  local SRV=$!
  local ok=0
  for i in $(seq 1 200); do
    grep -q "listening" "$TMP/mx3-$tag.log" 2>/dev/null && { ok=1; break; }
    sleep 2
  done
  if [ $ok -ne 1 ]; then
    echo "[$tag] STARTUP FAILED" | tee -a "$LOGF"; tail -3 "$TMP/mx3-$tag.log" | tee -a "$LOGF"
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; return 1
  fi
  local base peak v free_min
  base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  curl -s --noproxy '*' --max-time 1200 "http://127.0.0.1:9670/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"用三段话详细介绍大运河的历史与作用。"}],"n_predict":384,"temperature":0,"stream":false}' > "$TMPW/mx3-r.json" 2>/dev/null &
  local CURL=$!
  peak=$base; free_min=99999
  for i in $(seq 1 140); do
    kill -0 $CURL 2>/dev/null || break
    read -r v fv < <(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits)
    [ "$v" -gt "$peak" ] && peak=$v
    [ "$fv" -lt "$free_min" ] && free_min=$fv
    sleep 1
  done
  wait $CURL 2>/dev/null
  $PY "$TMP/mtx-extract.py" "$tag" "$TMPW/mx3-r.json" "$TMP/mx3-$tag.log" "峰值${peak}MiB(余${free_min}MiB)" "$M9" "${args[*]}" >> "$LOGF" 2>&1
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  sleep 3
}

echo "" >> "$LOGF"
echo "## 9B Q4_K_M 生产基准矩阵（config 复刻，72.38 锚点）" >> "$LOGF"
echo "| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |" >> "$LOGF"
echo "|---|---|---|---|---|---|" >> "$LOGF"

### 基准复刻（v26 / v25）
run_test "$BIN26" "v26-9B-replica"   "${BASE[@]}"
run_test "$BIN25" "v25-9B-replica"   "${BASE[@]}"

### KV 档横向
run_test "$BIN26" "v26-9B-q8q8"      "${BASE[@]}" -ctv q8_0
run_test "$BIN26" "v26-9B-t4t3"      "${BASE[@]}" -ctk turbo4 -ctv turbo3
run_test "$BIN26" "v26-9B-t4t2"      "${BASE[@]}" -ctk turbo4 -ctv turbo2
run_test "$BIN26" "v26-9B-f16"       "${BASE[@]}" -ctk f16 -ctv f16

### threads 对照（12 线程=hyperthread 关闭档）
run_test "$BIN26" "v26-9B-t12"       "${BASE[@]}" --threads 12

### ubatch 对照
run_test "$BIN26" "v26-9B-ub1024"    "${BASE[@]}" -ub 1024

### 视觉（mmproj CPU 卸载）
run_test "$BIN26" "v26-9B-vision"    "$MM9" "$TMPL_AG" $((9660+1)) "${BASE[@]}" --no-mmproj-offload

echo "=== 9B Q4KM MATRIX DONE $(date +%H:%M:%S) ===" | tee -a "$LOGF"

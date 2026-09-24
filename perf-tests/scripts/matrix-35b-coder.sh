#!/bin/bash
# 35B 生产基准矩阵：按 config.yaml「AgentWorld-mtp-Q80-128K-CODER」方案在 v26/v25 上复刻 + 增量实验
PY=/c/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe
BIN26=G:/Agents/kvmem-works/laamaafung/build-v26/bin/llama-server.exe
BIN25=G:/Agents/kvmem-works/wt-v25/build-v25/bin/llama-server.exe
M35="C:/WorkModels/Qwen3.6-35B-A3B/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf"
TMPL_AG="C:/iStartModel/tmpl/Qwen-Agentic-HONT.jinja"
TMP=/g/Agents/kvmem-works/_tmp
TMPW=G:/Agents/kvmem-works/_tmp
LOGF="$1"

# CODER 方案基准参数（config.yaml 复刻）
BASE=(-ngl all -ngld all --n-cpu-moe 34 --threads 18 --threads-http 2 --parallel 1 --kv-unified \
      -ctk q8_0 -ctv q8_0 -b 16384 -ub 256 --ctx-checkpoints 64 --load-mode mlock-ram --no-mmproj \
      --cache-prompt --cache-ram 8192 --fit on --spec-type draft-mtp --spec-draft-n-max 4 -c 131072)

run_test() {
  local BIN="$1"; shift
  local tag="$1"; shift
  local -a args=("$@")
  echo "=== [$tag] start $(date +%H:%M:%S) ===" | tee -a "$LOGF"
  "$BIN" -m "$M35" --chat-template-file "$TMPL_AG" "${args[@]}" --port 9660 > "$TMP/mx2-$tag.log" 2>&1 &
  local SRV=$!
  local ok=0
  for i in $(seq 1 200); do
    grep -q "listening" "$TMP/mx2-$tag.log" 2>/dev/null && { ok=1; break; }
    sleep 2
  done
  if [ $ok -ne 1 ]; then
    echo "[$tag] STARTUP FAILED" | tee -a "$LOGF"; tail -3 "$TMP/mx2-$tag.log" | tee -a "$LOGF"
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; return 1
  fi
  local base peak v free_min
  base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ",")
  curl -s --noproxy '*' --max-time 1200 "http://127.0.0.1:9660/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"用三段话详细介绍大运河的历史与作用。"}],"n_predict":384,"temperature":0,"stream":false}' > "$TMPW/mx2-r.json" 2>/dev/null &
  local CURL=$!
  peak=$base; free_min=99999
  for i in $(seq 1 140); do
    kill -0 $CURL 2>/dev/null || break
    read -r v fv < <(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | tr -d ",")
    [ "$v" -gt "$peak" ] && peak=$v
    [ "$fv" -lt "$free_min" ] && free_min=$fv
    sleep 1
  done
  wait $CURL 2>/dev/null
  local res
  res=$($PY "$TMP/mtx-extract.py" "$tag" "$TMPW/mx2-r.json" "$TMP/mx2-$tag.log" "峰值${peak}MiB(余${free_min}MiB)" "$M35" "${args[*]}" 2>&1)
  echo "$res" >> "$LOGF"
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  sleep 3
}

echo "| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |" >> "$LOGF"
echo "|---|---|---|---|---|---|" >> "$LOGF"

### 基准：CODER 方案复刻（v26 / v25 对照）
run_test "$BIN26" "v26-35B-CODER-replica"      "${BASE[@]}"
run_test "$BIN25" "v25-35B-CODER-replica"      "${BASE[@]}"

### KV 档横向（其余同 CODER）
run_test "$BIN26" "v26-35B-CODER-q8t4"         "${BASE[@]}" -ctv turbo4
run_test "$BIN26" "v26-35B-CODER-t4t3"         "${BASE[@]}" -ctk turbo4 -ctv turbo3
run_test "$BIN26" "v26-35B-CODER-t4t2"         "${BASE[@]}" -ctk turbo4 -ctv turbo2
run_test "$BIN26" "v26-35B-CODER-f16"          "${BASE[@]}" -ctk f16 -ctv f16
run_test "$BIN26" "v26-35B-CODER-q50q41"       "${BASE[@]}" -ctk q5_0 -ctv q4_1

### n-cpu-moe 扫描（34 基准向两侧）
run_test "$BIN26" "v26-35B-CODER-ncm32"        "${BASE[@]}" --n-cpu-moe 32
run_test "$BIN26" "v26-35B-CODER-ncm30"        "${BASE[@]}" --n-cpu-moe 30
run_test "$BIN26" "v26-35B-CODER-ncm28"        "${BASE[@]}" --n-cpu-moe 28
run_test "$BIN26" "v26-35B-CODER-ncm26"        "${BASE[@]}" --n-cpu-moe 26

### MTP n_max 对照
run_test "$BIN26" "v26-35B-CODER-nmax2"        "${BASE[@]}" --spec-draft-n-max 2
run_test "$BIN26" "v26-35B-CODER-nmax3"        "${BASE[@]}" --spec-draft-n-max 3

### ubatch 对照（config 其他方案用 auto/1024）
run_test "$BIN26" "v26-35B-CODER-ub1024"       "${BASE[@]}" -ub 1024

### v25 对照：CODER + q8t4（生产常用 KV）
run_test "$BIN25" "v25-35B-CODER-q8t4"         "${BASE[@]}" -ctv turbo4
run_test "$BIN25" "v25-35B-CODER-nmax2"        "${BASE[@]}" --spec-draft-n-max 2

echo "=== 35B CODER MATRIX DONE $(date +%H:%M:%S) ===" | tee -a "$LOGF"

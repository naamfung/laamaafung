#!/bin/bash
# 27B 三值补充矩阵：PQ2_0 入组 + 256K KVMem 组（用户指出的两块缺口）
PY=/c/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe
BIN=G:/Agents/kvmem-works/laamaafung/build-v26/bin/llama-server.exe
M27Q="C:/WorkModels/Qwen3.8-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
M27T="C:/WorkModels/Qwen3.8-27B/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
TMPL="C:/WorkModels/Jinja-Template/qwen3.8-froggeric-v22.5.jinja"
TMP=/g/Agents/kvmem-works/_tmp
LOGF="$1"
run_test() {
  local MODEL="$1"; local tag="$2"; shift 2
  local -a args=("$@")
  echo "=== [$tag] start $(date +%H:%M:%S) ===" >> "$LOGF"
  "$BIN" -m "$MODEL" --chat-template-file "$TMPL" "${args[@]}" --port 9685 > "$TMP/mx5-$tag.log" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 240); do
    grep -q "listening" "$TMP/mx5-$tag.log" 2>/dev/null && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break
    sleep 2
  done
  if [ $ok -ne 1 ]; then
    local reason
    reason=$(grep -oiE "failed to fit[^\"]*|out of memory|error[^\"]*" "$TMP/mx5-$tag.log" | head -2 | tr '\n' ' ')
    echo "| $tag | - | - | - | ❌ 放不下/启动失败（$reason） | \`${args[*]}\` |" >> "$LOGF"
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    return 1
  fi
  local base peak v free_min
  base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ",")
  curl -s --noproxy '*' --max-time 1800 "http://127.0.0.1:9685/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"用三段话详细介绍大运河的历史与作用。"}],"n_predict":4096,"temperature":0.6,"stream":false}' > "$TMP/mx5-r.json" 2>/dev/null &
  CURL=$!
  peak=$base; free_min=99999
  for i in $(seq 1 1900); do
    kill -0 $CURL 2>/dev/null || break
    read -r v fv < <(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | tr -d ",")
    [ "$v" -gt "$peak" ] && peak=$v
    [ "$fv" -lt "$free_min" ] && free_min=$fv
    sleep 1
  done
  wait $CURL 2>/dev/null
  $PY "$TMP/mtx-extract.py" "$tag" "$TMP/mx5-r.json" "$TMP/mx5-$tag.log" "峰值${peak}MiB(余${free_min}MiB)" "$MODEL" "${args[*]}" >> "$LOGF" 2>&1
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  sleep 3
}
echo "" >> "$LOGF"
echo "## 27B 三值补充段（PQ2_0 入组 + 256K KVMem；n_predict 4096 生产口径）" >> "$LOGF"
echo "| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |" >> "$LOGF"
echo "|---|---|---|---|---|---|" >> "$LOGF"
# --- PQ2_0 段 ---
run_test "$M27Q" "v26-27B-PQ2_0-kvmem4K"        -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$M27Q" "v26-27B-PQ2_0-kvmem4K-turbo4" -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$M27Q" "v26-27B-PQ2_0-kvmem-256K"     -ngl 99 -c 262144 -fit off -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 8192 --kvmem-gen-reserve 2048
# --- PTQ1_0 256K 段 ---
run_test "$M27T" "v26-27B-PTQ1_0-kvmem-256K"    -ngl 99 -c 262144 -fit off -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 8192 --kvmem-gen-reserve 2048
run_test "$M27T" "v26-27B-PTQ1_0-plain-256K"    -ngl 99 -c 262144 -fit off -ctk q8_0 -ctv turbo4
echo "=== 27B SUPPLEMENT DONE $(date +%H:%M:%S) ===" >> "$LOGF"

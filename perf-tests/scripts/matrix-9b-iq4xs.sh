#!/bin/bash
# 9B IQ4_XS-UP 补充矩阵（对齐开发期 80-87 t/s 基线的量化）
PY=/c/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe
BIN26=G:/Agents/kvmem-works/laamaafung/build-v26/bin/llama-server.exe
BIN25=G:/Agents/kvmem-works/wt-v25/build-v25/bin/llama-server.exe
TMP=/g/Agents/kvmem-works/_tmp

M9="C:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-Genesis-FITKIT-IQ4_XS-UP-4.888G-genesis-imatrix/Qwen3.5-9B-Uncensored-Genesis-FITKIT-IQ4_XS-UP-4.888G-genesis-imatrix.gguf"
MM9="C:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
TMPL_AG="C:/iStartModel/tmpl/Qwen-Agentic-HONT.jinja"

LOGF="$1"

run_test() {
  local BIN="$1"; shift
  local tag="$1"; shift
  local model="$1"; shift
  local mm="$1"; shift
  local tmpl="$1"; shift
  local port="$1"; shift
  local -a args=("$@")
  local margs=()
  [ "$mm" != "-" ] && margs+=(-mm "$mm" --no-mmproj-offload)
  [ "$tmpl" != "-" ] && margs+=(--chat-template-file "$tmpl")
  echo "=== [$tag] start $(date +%H:%M:%S) ===" | tee -a "$LOGF"
  "$BIN" -m "$model" "${margs[@]}" "${args[@]}" --port "$port" > "$TMP/mtx-$tag.log" 2>&1 &
  local SRV=$!
  local ok=0
  for i in $(seq 1 120); do
    grep -q "listening" "$TMP/mtx-$tag.log" 2>/dev/null && { ok=1; break; }
    sleep 2
  done
  if [ $ok -ne 1 ]; then
    echo "[$tag] STARTUP FAILED" | tee -a "$LOGF"; tail -3 "$TMP/mtx-$tag.log" | tee -a "$LOGF"
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; return 1
  fi
  local body1 body2
  if [ "$mm" != "-" ]; then
    local b64
    b64=$($PY -c "import base64;print(base64.b64encode(open(r'G:/Agents/kvmem-works/.workbuddy/tmp/img_red.png','rb').read()).decode())")
    body1="{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$b64\"}},{\"type\":\"text\",\"text\":\"这张图片的主要颜色是什么？详细说明。\"}]}],\"n_predict\":192,\"temperature\":0,\"stream\":false}"
    body2="$body1"
  else
    body1="{\"messages\":[{\"role\":\"user\",\"content\":\"用三段话详细介绍大运河的历史与作用。\"}],\"n_predict\":384,\"temperature\":0,\"stream\":false}"
    body2="$body1"
  fi
  curl -s --noproxy '*' --max-time 900 "http://127.0.0.1:$port/v1/chat/completions" -H "Content-Type: application/json" -d "$body1" > "$TMPW/mtx-r1.json" 2>/dev/null
  curl -s --noproxy '*' --max-time 900 "http://127.0.0.1:$port/v1/chat/completions" -H "Content-Type: application/json" -d "$body2" > "$TMPW/mtx-r2.json" 2>/dev/null
  local vram
  vram="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d "," 2>/dev/null | head -1)/余$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)MiB"
  $PY "$TMP/mtx-extract.py" "$tag" "$TMPW/mtx-r2.json" "$TMP/mtx-$tag.log" "$vram" "$model" "${args[*]}" >> "$LOGF" 2>&1
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  sleep 3
}

echo "" >> "$LOGF"
echo "## 9B IQ4_XS-UP 补充（对齐开发期基线量化）" >> "$LOGF"
echo "| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |" >> "$LOGF"
echo "|---|---|---|---|---|---|" >> "$LOGF"

PORT=9450
run_test "$BIN26" "v26-9Bi-base"      "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0
run_test "$BIN26" "v26-9Bi-f16"       "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk f16 -ctv f16
run_test "$BIN26" "v26-9Bi-turbo4"    "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo4
run_test "$BIN26" "v26-9Bi-turbo3"    "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo3
run_test "$BIN26" "v26-9Bi-turbo2"    "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo2
run_test "$BIN26" "v26-9Bi-tq3"       "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo3_tcq
run_test "$BIN26" "v26-9Bi-tq2"       "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo2_tcq
run_test "$BIN26" "v26-9Bi-turbo1.5"  "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo1.5
run_test "$BIN26" "v26-9Bi-kvarn4"    "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk kvarn8 -ctv kvarn4
run_test "$BIN26" "v26-9Bi-kvarn8-3"  "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk kvarn8 -ctv kvarn3
run_test "$BIN26" "v26-9Bi-kvarn4-3"  "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk kvarn4 -ctv kvarn3
run_test "$BIN26" "v26-9Bi-kvarn3-2"  "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk kvarn3 -ctv kvarn2
run_test "$BIN26" "v26-9Bi-kvarn44"   "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk kvarn4 -ctv kvarn4
run_test "$BIN26" "v26-9Bi-turbo4-3"  "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk turbo4 -ctv turbo3
run_test "$BIN26" "v26-9Bi-turbo3-2"  "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk turbo3 -ctv turbo2
run_test "$BIN26" "v26-9Bi-kvmem"     "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN26" "v26-9Bi-kvmem-turbo4" "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN26" "v26-9Bi-kvmem-kvarn4" "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk kvarn8 -ctv kvarn4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN26" "v26-9Bi-vision"    "$M9" "$MM9" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo4
# v25 对等
run_test "$BIN25" "v25-9Bi-base"      "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0
run_test "$BIN25" "v25-9Bi-f16"       "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk f16 -ctv f16
run_test "$BIN25" "v25-9Bi-turbo4"    "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo4
run_test "$BIN25" "v25-9Bi-turbo3"    "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo3
run_test "$BIN25" "v25-9Bi-turbo2"    "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo2
run_test "$BIN25" "v25-9Bi-tq3"       "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo3_tcq
run_test "$BIN25" "v25-9Bi-kvmem"     "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN25" "v25-9Bi-kvmem-turbo4" "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN25" "v25-9Bi-vision"    "$M9" "$MM9" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo4
echo "=== 9B IQ4_XS SUPPLEMENT DONE $(date +%H:%M:%S) ===" | tee -a "$LOGF"

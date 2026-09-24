#!/bin/bash
# KVarN 性能矩阵（双引擎对比）：v26（KVarN 引擎）vs v25（基线）
# 对等配置两引擎各测一组；KVarN 系仅 v26。用法: bash matrix-test.sh <results.md>
PY=/c/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe
BIN26=G:/Agents/kvmem-works/laamaafung/build-v26/bin/llama-server.exe
BIN25=G:/Agents/kvmem-works/wt-v25/build-v25/bin/llama-server.exe
TMP=/g/Agents/kvmem-works/_tmp
TMPW=G:/Agents/kvmem-works/_tmp
mkdir -p "$TMP/matrix"

M9="C:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-Genesis-FITKIT-Q3_K_L-DOWN-4.88G-genesis-imatrix/Qwen3.5-9B-Uncensored-Genesis-FITKIT-Q3_K_L-DOWN-4.88G-genesis-imatrix.gguf"
MM9="C:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
M35="C:/WorkModels/Qwen3.6-35B-A3B/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf"
MM35="C:/WorkModels/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-mmproj-BF16.gguf"
M27="C:/WorkModels/Qwen3.8-27B/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
M27Q2="C:/WorkModels/Qwen3.8-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
TMPL_AG="C:/iStartModel/tmpl/Qwen-Agentic-HONT.jinja"
TMPL_FR="C:/WorkModels/Jinja-Template/qwen3.8-froggeric-v22.5.jinja"

LOGF="$1"

# run_test <bin> <tag> <model> <mmproj|-> <template|-> <port> <extra-args...>
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
  "$BIN" -m "$model" "${margs[@]}" "${args[@]}" --port "$port" \
      > "$TMP/mtx-$tag.log" 2>&1 &
  local SRV=$!
  local ok=0
  for i in $(seq 1 120); do
    grep -q "listening" "$TMP/mtx-$tag.log" 2>/dev/null && { ok=1; break; }
    grep -qiE "error.*failed|llama_init_from_model.*null" "$TMP/mtx-$tag.log" 2>/dev/null && break
    sleep 2
  done
  if [ $ok -ne 1 ]; then
    echo "[$tag] STARTUP FAILED" | tee -a "$LOGF"
    tail -5 "$TMP/mtx-$tag.log" | tee -a "$LOGF"
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    return 1
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

  curl -s --noproxy '*' --max-time 900 "http://127.0.0.1:$port/v1/chat/completions" -H "Content-Type: application/json" -d "$body1" > "$TMP/mtx-r1.json" 2>/dev/null
  curl -s --noproxy '*' --max-time 900 "http://127.0.0.1:$port/v1/chat/completions" -H "Content-Type: application/json" -d "$body2" > "$TMPW/mtx-r2.json" 2>/dev/null

  local vram
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)

  $PY "$TMP/mtx-extract.py" "$tag" "$TMPW/mtx-r2.json" "$TMP/mtx-$tag.log" "$vram" "$model" "${args[*]}" >> "$LOGF" 2>&1

  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  sleep 3
}

cat > "$TMP/mtx-extract.py" << 'PYEOF'
import json, sys, re
tag, rj, slog, vram, model, args = sys.argv[1:7]
try:
    d = json.load(open(rj, encoding='utf-8'))
    t = d.get('timings', d.get('usage', {}))
except Exception as e:
    print(f"| {tag} | 请求失败: {e} | | | | |")
    sys.exit(0)
pps = t.get('prompt_per_second')
gps = t.get('predicted_per_second')
s = open(slog, encoding='utf-8', errors='replace').read()
m3 = re.findall(r'tg_3s = +([0-9.]+)', s)
tg3 = max(float(x) for x in m3) if m3 else ''
acc = re.findall(r'draft acceptance = +([0-9.]+).*?mean len = +([0-9.]+)', s)
acctxt = f"{acc[-1][0]}/{acc[-1][1]}" if acc else ''
print(f"| {tag} | gen {gps:.1f} t/s (tg_3s峰值 {tg3}) | prefill {pps:.0f} t/s | {acctxt} | {vram} MB | `{args}` |")
PYEOF

echo "| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |" >> "$LOGF"
echo "|---|---|---|---|---|---|" >> "$LOGF"

PORT=9200

### ================= 9B（Q3_K_L 全GPU, 无MTP, 有mmproj） =================
# --- v26 ---
run_test "$BIN26" "v26-9B-base"        "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0
run_test "$BIN26" "v26-9B-f16"         "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk f16 -ctv f16
run_test "$BIN26" "v26-9B-turbo4"      "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo4
run_test "$BIN26" "v26-9B-turbo3"      "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo3
run_test "$BIN26" "v26-9B-turbo2"      "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo2
run_test "$BIN26" "v26-9B-tq3"         "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo3_tcq
run_test "$BIN26" "v26-9B-tq2"         "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo2_tcq
run_test "$BIN26" "v26-9B-turbo1.5"    "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo1.5
run_test "$BIN26" "v26-9B-kvarn4"      "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk kvarn8 -ctv kvarn4
run_test "$BIN26" "v26-9B-kvarn8-3"    "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk kvarn8 -ctv kvarn3
run_test "$BIN26" "v26-9B-kvarn4-3"    "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk kvarn4 -ctv kvarn3
run_test "$BIN26" "v26-9B-kvarn3-2"    "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk kvarn3 -ctv kvarn2
run_test "$BIN26" "v26-9B-kvarn44"     "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk kvarn4 -ctv kvarn4
run_test "$BIN26" "v26-9B-turbo4-3"    "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk turbo4 -ctv turbo3
run_test "$BIN26" "v26-9B-turbo3-2"    "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk turbo3 -ctv turbo2
run_test "$BIN26" "v26-9B-kvmem"       "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN26" "v26-9B-kvmem-turbo4" "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN26" "v26-9B-kvmem-kvarn4" "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk kvarn8 -ctv kvarn4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN26" "v26-9B-vision"      "$M9" "$MM9" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo4
# --- v25 对等（无 kvarn 系） ---
run_test "$BIN25" "v25-9B-base"        "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0
run_test "$BIN25" "v25-9B-f16"         "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk f16 -ctv f16
run_test "$BIN25" "v25-9B-turbo4"      "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo4
run_test "$BIN25" "v25-9B-turbo3"      "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo3
run_test "$BIN25" "v25-9B-turbo2"      "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo2
run_test "$BIN25" "v25-9B-tq3"         "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo3_tcq
run_test "$BIN25" "v25-9B-tq2"         "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo2_tcq
run_test "$BIN25" "v25-9B-turbo1.5"    "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo1.5
run_test "$BIN25" "v25-9B-kvmem"       "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN25" "v25-9B-kvmem-turbo4" "$M9" "-" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN25" "v25-9B-vision"      "$M9" "$MM9" "$TMPL_AG" $((PORT++)) -ngl 99 -c 8192 -ctk q8_0 -ctv turbo4

### ================= 35B（MoE 17G, n-cpu-moe 36, MTP, mmproj） =================
MOE="--n-cpu-moe 36"
# --- v26 ---
run_test "$BIN26" "v26-35B-base"       "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo4
run_test "$BIN26" "v26-35B-mtp"        "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp
run_test "$BIN26" "v26-35B-mtp-nmax2"  "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp --spec-draft-n-max 2
run_test "$BIN26" "v26-35B-mtp-turbo3" "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo3 --spec-type draft-mtp
run_test "$BIN26" "v26-35B-mtp-turbo2" "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo2 --spec-type draft-mtp
run_test "$BIN26" "v26-35B-mtp-tq3"    "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo3_tcq --spec-type draft-mtp
run_test "$BIN26" "v26-35B-mtp-tq3-nmax2" "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo3_tcq --spec-type draft-mtp --spec-draft-n-max 2
run_test "$BIN26" "v26-35B-mtp-kvarn4" "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk kvarn8 -ctv kvarn4 --spec-type draft-mtp
run_test "$BIN26" "v26-35B-mtp-kvarn4-nmax2" "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk kvarn8 -ctv kvarn4 --spec-type draft-mtp --spec-draft-n-max 2
run_test "$BIN26" "v26-35B-mtp-kvarn8-3" "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk kvarn8 -ctv kvarn3 --spec-type draft-mtp
run_test "$BIN26" "v26-35B-mtp-kvarn3-2" "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk kvarn3 -ctv kvarn2 --spec-type draft-mtp --spec-draft-n-max 2
run_test "$BIN26" "v26-35B-mtp-turbo3-2" "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk turbo3 -ctv turbo2 --spec-type draft-mtp
run_test "$BIN26" "v26-35B-mtp-kvmem"  "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN26" "v26-35B-mtp-kvmem-nmax2" "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp --spec-draft-n-max 2 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN26" "v26-35B-mtp-kvmem-tq3" "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo3_tcq --spec-type draft-mtp --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN26" "v26-35B-vision-mtp" "$M35" "$MM35" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp
# --- v25 对等 ---
run_test "$BIN25" "v25-35B-base"       "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo4
run_test "$BIN25" "v25-35B-mtp"        "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp
run_test "$BIN25" "v25-35B-mtp-nmax2"  "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp --spec-draft-n-max 2
run_test "$BIN25" "v25-35B-mtp-turbo3" "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo3 --spec-type draft-mtp
run_test "$BIN25" "v25-35B-mtp-turbo2" "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo2 --spec-type draft-mtp
run_test "$BIN25" "v25-35B-mtp-tq3"    "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo3_tcq --spec-type draft-mtp
run_test "$BIN25" "v25-35B-mtp-kvmem"  "$M35" "-" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN25" "v25-35B-vision-mtp" "$M35" "$MM35" "$TMPL_AG" $((PORT++)) -ngl 99 $MOE -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp

### ================= 27B 三值 PQ2_0（全GPU, 无MTP无mmproj） =================
# --- v26 ---
run_test "$BIN26" "v26-27B-base"       "$M27" "-" "$TMPL_FR" $((PORT++)) -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0
run_test "$BIN26" "v26-27B-turbo4"     "$M27" "-" "$TMPL_FR" $((PORT++)) -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo4
run_test "$BIN26" "v26-27B-turbo3"     "$M27" "-" "$TMPL_FR" $((PORT++)) -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo3
run_test "$BIN26" "v26-27B-turbo2"     "$M27" "-" "$TMPL_FR" $((PORT++)) -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo2
run_test "$BIN26" "v26-27B-turbo3-2"   "$M27" "-" "$TMPL_FR" $((PORT++)) -ngl 99 -c 4096 -ctk turbo3 -ctv turbo2
run_test "$BIN26" "v26-27B-kvarn4"     "$M27" "-" "$TMPL_FR" $((PORT++)) -ngl 99 -c 4096 -ctk kvarn8 -ctv kvarn4
run_test "$BIN26" "v26-27B-kvarn3-2"   "$M27" "-" "$TMPL_FR" $((PORT++)) -ngl 99 -c 4096 -ctk kvarn3 -ctv kvarn2
run_test "$BIN26" "v26-27B-kvmem"      "$M27" "-" "$TMPL_FR" $((PORT++)) -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN26" "v26-27B-kvmem-turbo4" "$M27" "-" "$TMPL_FR" $((PORT++)) -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN26" "v26-27B-pq2-kvmem"   "$M27Q2" "-" "$TMPL_FR" $((PORT++)) -ngl 99 -c 2048 -fit off -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
# --- v25 对等（同时验证 v25 三值修复） ---
run_test "$BIN25" "v25-27B-base"       "$M27" "-" "$TMPL_FR" $((PORT++)) -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0
run_test "$BIN25" "v25-27B-turbo4"     "$M27" "-" "$TMPL_FR" $((PORT++)) -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo4
run_test "$BIN25" "v25-27B-kvmem"      "$M27" "-" "$TMPL_FR" $((PORT++)) -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN25" "v25-27B-pq2-kvmem" "$M27Q2" "-" "$TMPL_FR" $((PORT++)) -ngl 99 -c 2048 -fit off -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024

echo "=== MATRIX DONE $(date +%H:%M:%S) ===" | tee -a "$LOGF"

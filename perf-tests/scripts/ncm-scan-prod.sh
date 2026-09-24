#!/bin/bash
# 生产口径 n-cpu-moe 极限扫描：128K(无KVMem) 与 256K(KVMem 32K/16K)
PY=/c/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe
BIN26=G:/Agents/kvmem-works/laamaafung/build-v26/bin/llama-server.exe
M35="C:/WorkModels/Qwen3.6-35B-A3B/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf"
TMP=/g/Agents/kvmem-works/_tmp
TMPW=G:/Agents/kvmem-works/_tmp
LOGF="$1"

run_scan() {
  local mode="$1"; local ctx="$2"; shift 2
  local -a extra=("$@")
  echo "" >> "$LOGF"
  echo "## 生产口径扫描 [$mode]（ctx=$ctx, MTP + turbo4 + q8_0 K）" >> "$LOGF"
  echo "| n-cpu-moe | 显存峰值(余量) | gen t/s (tg_3s峰值) | 状态 |" >> "$LOGF"
  echo "|---|---|---|---|" >> "$LOGF"
  local NCM=36
  while [ $NCM -ge 24 ]; do
    "$BIN26" -m "$M35" -ngl 99 --n-cpu-moe $NCM --parallel 1 -c $ctx -ctk q8_0 -ctv turbo4 --spec-type draft-mtp \
        "${extra[@]}" --port 9590 > "$TMP/ncmp-$mode-$NCM.log" 2>&1 &
    local SRV=$!
    local ok=0
    for i in $(seq 1 150); do
      grep -q "listening" "$TMP/ncmp-$mode-$NCM.log" 2>/dev/null && { ok=1; break; }
      sleep 2
    done
    if [ $ok -ne 1 ]; then
      local reason
      reason=$(grep -oiE "failed to fit[^\"]*|out of memory" "$TMP/ncmp-$mode-$NCM.log" | head -1)
      echo "| $NCM | - | - | ❌ 放不下（$reason） |" >> "$LOGF"
      kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
      echo "[$mode] n-cpu-moe=$NCM 到达极限，扫描结束" | tee -a "$LOGF"
      break
    fi
    local base peak v free_min
    base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ",")
    curl -s --noproxy '*' --max-time 900 "http://127.0.0.1:9590/v1/chat/completions" -H "Content-Type: application/json" \
      -d '{"messages":[{"role":"user","content":"用三段话详细介绍大运河的历史与作用。"}],"n_predict":384,"temperature":0,"stream":false}' > "$TMPW/ncmp-r.json" 2>/dev/null &
    local CURL=$!
    peak=$base; free_min=99999
    for i in $(seq 1 100); do
      kill -0 $CURL 2>/dev/null || break
      read -r v fv < <(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | tr -d ",")
      [ "$v" -gt "$peak" ] && peak=$v
      [ "$fv" -lt "$free_min" ] && free_min=$fv
      sleep 1
    done
    wait $CURL 2>/dev/null
    local res
    res=$($PY "$TMP/ncm-extract.py" "$TMPW/ncmp-r.json" "$TMP/ncmp-$mode-$NCM.log" 2>&1)
    echo "| $NCM | 峰值${peak}MiB(余${free_min}MiB) | $res |  |" >> "$LOGF"
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    sleep 3
    NCM=$((NCM-2))
  done
}

run_scan "128K-plain" 131072
run_scan "256K-kvmem32K" 262144 --kvmem --kvmem-budget 32768 --kvmem-gen-reserve 16384
echo "=== PROD SCAN DONE $(date +%H:%M:%S) ===" | tee -a "$LOGF"

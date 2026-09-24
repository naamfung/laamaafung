#!/bin/bash
# n-cpu-moe × KVMem 池宽 交叉矩阵：层数（36→24）与池宽（plain/8K/16K/32K）两维独立扫描
# 单组合 fit 失败只记录该格，不终止（更小层可能恢复放得下），28 组合全覆盖
PY=/c/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe
BIN26=G:/Agents/kvmem-works/laamaafung/build-v26/bin/llama-server.exe
M35="C:/WorkModels/Qwen3.6-35B-A3B/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf"
TMP=/g/Agents/kvmem-works/_tmp
TMPW=G:/Agents/kvmem-works/_tmp
LOGF="$1"

# 池宽定义：名称 参数...
POOLS=("plain|" "kvmem-8K-4K|--kvmem --kvmem-budget 8192 --kvmem-gen-reserve 4096" \
       "kvmem-16K-8K|--kvmem --kvmem-budget 16384 --kvmem-gen-reserve 8192" \
       "kvmem-32K-16K|--kvmem --kvmem-budget 32768 --kvmem-gen-reserve 16384")

echo "" >> "$LOGF"
echo "## n-cpu-moe × KVMem 池宽 交叉矩阵（35B + MTP + turbo4，两维独立全组合）" >> "$LOGF"
echo "| 层数 \\ 池宽 | plain | 8K/4K | 16K/8K | 32K/16K |" >> "$LOGF"
echo "|---|---|---|---|---|" >> "$LOGF"

for NCM in 36 34 32 30 28 26 24; do
  row="| $NCM "
  for pool in "${POOLS[@]}"; do
    pname="${pool%%|*}"
    pargs="${pool#*|}"
    declare -a karr=()
    [ -n "$pargs" ] && read -ra karr <<< "$pargs"
    "$BIN26" -m "$M35" -ngl 99 --n-cpu-moe $NCM -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp \
        "${karr[@]}" --port 9580 > "$TMP/ncm-$pname-$NCM.log" 2>&1 &
    SRV=$!
    ok=0
    for i in $(seq 1 100); do
      grep -q "listening" "$TMP/ncm-$pname-$NCM.log" 2>/dev/null && { ok=1; break; }
      sleep 2
    done
    if [ $ok -ne 1 ]; then
      row="$row | ❌ 放不下"
      kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
      continue
    fi
    base peak v free_min free_load
    base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    free_load=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    curl -s --noproxy '*' --max-time 900 "http://127.0.0.1:9580/v1/chat/completions" -H "Content-Type: application/json" \
      -d '{"messages":[{"role":"user","content":"用三段话详细介绍大运河的历史与作用。"}],"n_predict":384,"temperature":0,"stream":false}' > "$TMPW/ncm-r.json" 2>/dev/null &
    CURL=$!
    peak=$base
    free_min=99999
    for i in $(seq 1 100); do
      kill -0 $CURL 2>/dev/null || break
      read -r v fv < <(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits)
      [ "$v" -gt "$peak" ] && peak=$v
      [ "$fv" -lt "$free_min" ] && free_min=$fv
      sleep 1
    done
    wait $CURL 2>/dev/null
    res
    res=$($PY "$TMP/ncm-extract.py" "$TMPW/ncm-r.json" "$TMP/ncm-$pname-$NCM.log" 2>&1)
    row="$row | 峰值${peak}MiB(余${free_min}MiB,载后余${free_load}MiB) $res"
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    sleep 3
  done
  echo "$row |" >> "$LOGF"
  echo "层数 $NCM 行完成 $(date +%H:%M:%S)" | tee -a "$LOGF"
done
echo "=== NCM×POOL MATRIX DONE $(date +%H:%M:%S) ===" | tee -a "$LOGF"

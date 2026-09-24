#!/bin/bash
# 显存验证：起 server → 短请求期间抓显存峰值 → 杀
# 用法: bash vram-check.sh <tag> <bin> <model-args...>
TAG="$1"; shift
BIN="$1"; shift
TMP=/g/Agents/kvmem-works/_tmp
TMPW=G:/Agents/kvmem-works/_tmp
PORT=9350

"$BIN" "$@" --port $PORT > "$TMP/vr-$TAG.log" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 150); do
  grep -q "listening" "$TMP/vr-$TAG.log" 2>/dev/null && { ok=1; break; }
  grep -qiE "error|failed to|null" "$TMP/vr-$TAG.log" 2>/dev/null && break
  sleep 2
done
if [ $ok -ne 1 ]; then
  echo "[$TAG] STARTUP FAILED:"; tail -4 "$TMP/vr-$TAG.log"
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; exit 1
fi

# 加载完基线显存
base_vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

# 请求放后台，同时抓峰值
curl -s --noproxy '*' --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"用三段话详细介绍大运河的历史与作用。"}],"n_predict":4096,"temperature":0.6,"stream":false}' \
  > "$TMPW/vr-$TAG-resp.json" 2>/dev/null &
CURL=$!
peak=$base_vram
for i in $(seq 1 60); do
  kill -0 $CURL 2>/dev/null || break
  v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$v" -gt "$peak" ] && peak=$v
  sleep 1
done
wait $CURL 2>/dev/null

# 请求结果摘要 + 速度
result=$(/c/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe -c "
import json
try:
    d = json.load(open(r'$TMPW/vr-$TAG-resp.json', encoding='utf-8'))
    t = d['timings']
    print(f\"gen {t['predicted_per_second']:.1f} t/s, prefill {t['prompt_per_second']:.0f} t/s\")
except Exception as e:
    print('RESP FAIL', e)
" 2>/dev/null)

echo "[$TAG] 加载后=${base_vram}MiB 峰值=${peak}MiB / 8188MiB | $result"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
sleep 3

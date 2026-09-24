#!/bin/bash
PY=/c/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe
BIN26=G:/Agents/kvmem-works/laamaafung/build-v26/bin/llama-server.exe
BIN25=G:/Agents/kvmem-works/wt-v25/build-v25/bin/llama-server.exe
M27="C:/WorkModels/Qwen3.8-27B/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
TMPL_FR="C:/WorkModels/Jinja-Template/qwen3.8-froggeric-v22.5.jinja"
TMP=/g/Agents/kvmem-works/_tmp
TMPW=G:/Agents/kvmem-works/_tmp
LOGF="$1"
run_test() {
  local BIN="$1"; shift
  local tag="$1"; shift
  local -a args=("$@")
  echo "=== [$tag] start $(date +%H:%M:%S) ===" >> "$LOGF"
  "$BIN" -m "$M27" --chat-template-file "$TMPL_FR" "${args[@]}" --port 9685 > "$TMP/mx4-$tag.log" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 200); do
    grep -q "listening" "$TMP/mx4-$tag.log" 2>/dev/null && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break
    sleep 2
  done
  if [ $ok -ne 1 ]; then
    echo "[$tag] STARTUP FAILED" >> "$LOGF"; tail -3 "$TMP/mx4-$tag.log" >> "$LOGF"
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; return 1
  fi
  base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ",")
  curl -s --noproxy '*' --max-time 900 "http://127.0.0.1:9685/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"用三段话详细介绍大运河的历史与作用。"}],"n_predict":384,"temperature":0,"stream":false}' > "$TMPW/mx4-r.json" 2>/dev/null &
  CURL=$!
  peak=$base; free_min=99999
  for i in $(seq 1 140); do
    kill -0 $CURL 2>/dev/null || break
    read -r v fv < <(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | tr -d ",")
    [ "$v" -gt "$peak" ] && peak=$v
    [ "$fv" -lt "$free_min" ] && free_min=$fv
    sleep 1
  done
  wait $CURL 2>/dev/null
  $PY "$TMP/mtx-extract.py" "$tag" "$TMPW/mx4-r.json" "$TMP/mx4-$tag.log" "峰值${peak}MiB(余${free_min}MiB)" "$M27" "${args[*]}" >> "$LOGF" 2>&1
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  sleep 3
}
echo "" >> "$LOGF"
echo "## 27B 三值 PTQ1_0 段（-c 4096 -fit off）" >> "$LOGF"
echo "| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |" >> "$LOGF"
echo "|---|---|---|---|---|---|" >> "$LOGF"
run_test "$BIN26" "v26-27B-base"        -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0
run_test "$BIN26" "v26-27B-turbo4"      -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo4
run_test "$BIN26" "v26-27B-turbo3"      -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo3
run_test "$BIN26" "v26-27B-turbo2"      -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo2
run_test "$BIN26" "v26-27B-turbo3-2"    -ngl 99 -c 4096 -fit off -ctk turbo3 -ctv turbo2
run_test "$BIN26" "v26-27B-kvarn4"      -ngl 99 -c 4096 -fit off -ctk kvarn8 -ctv kvarn4
run_test "$BIN26" "v26-27B-kvarn3-2"    -ngl 99 -c 4096 -fit off -ctk kvarn3 -ctv kvarn2
run_test "$BIN26" "v26-27B-kvmem"       -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN26" "v26-27B-kvmem-turbo4" -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
run_test "$BIN25" "v25-27B-base"        -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0
run_test "$BIN25" "v25-27B-turbo4"      -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo4
run_test "$BIN25" "v25-27B-kvmem"       -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024
echo "=== 27B SEGMENT DONE $(date +%H:%M:%S) ===" >> "$LOGF"

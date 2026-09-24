#!/bin/bash
# turbo1.5 KV 补测：9B 对称/反向 + 27B + 35B（n_predict 4096 生产口径）
PY=/c/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe
BIN=G:/Agents/kvmem-works/laamaafung/build-v26/bin/llama-server.exe
M9="C:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-Genesis-FITKIT-Q3_K_L-DOWN-4.88G-genesis-imatrix/Qwen3.5-9B-Uncensored-Genesis-FITKIT-Q3_K_L-DOWN-4.88G-genesis-imatrix.gguf"
M27="C:/WorkModels/Qwen3.8-27B/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
M35="C:/WorkModels/Qwen3.6-35B-A3B/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf"
TMPL9="C:/iStartModel/tmpl/Qwen-Agentic-HONT.jinja"
TMPL27="C:/WorkModels/Jinja-Template/qwen3.8-froggeric-v22.5.jinja"
TMPL35="C:/iStartModel/tmpl/Qwen-Agentic-HONT.jinja"
TMP=/g/Agents/kvmem-works/_tmp
LOGF="$1"
run_test() {
  local MODEL="$1"; local TMPL="$2"; local tag="$3"; shift 3
  local -a args=("$@")
  echo "=== [$tag] start $(date +%H:%M:%S) ===" >> "$LOGF"
  "$BIN" -m "$MODEL" --chat-template-file "$TMPL" "${args[@]}" --port 9687 > "$TMP/t15-$tag.log" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 240); do
    grep -q "listening" "$TMP/t15-$tag.log" 2>/dev/null && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break
    sleep 2
  done
  if [ $ok -ne 1 ]; then
    local reason
    reason=$(grep -oiE "failed to fit[^\"]*|out of memory|fatal error[^\"]*" "$TMP/t15-$tag.log" | head -1)
    echo "| $tag | - | - | - | ❌ 启动失败（$reason） | \`${args[*]}\` |" >> "$LOGF"
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    return 1
  fi
  local base peak v free_min
  base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ",")
  curl -s --noproxy '*' --max-time 1200 "http://127.0.0.1:9687/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"用三段话详细介绍大运河的历史与作用。"}],"n_predict":4096,"temperature":0.6,"stream":false}' > "$TMP/t15-r.json" 2>/dev/null &
  CURL=$!
  peak=$base; free_min=99999
  for i in $(seq 1 1300); do
    kill -0 $CURL 2>/dev/null || break
    read -r v fv < <(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | tr -d ",")
    [ "$v" -gt "$peak" ] && peak=$v
    [ "$fv" -lt "$free_min" ] && free_min=$fv
    sleep 1
  done
  wait $CURL 2>/dev/null
  $PY "$TMP/mtx-extract.py" "$tag" "$TMP/t15-r.json" "$TMP/t15-$tag.log" "峰值${peak}MiB(余${free_min}MiB)" "$MODEL" "${args[*]}" >> "$LOGF" 2>&1
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  sleep 3
}
echo "" >> "$LOGF"
echo "## turbo1.5 KV 补测（n_predict 4096 生产口径；9B 曾测过 q8_0×turbo1.5=59.2 不重复）" >> "$LOGF"
echo "| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |" >> "$LOGF"
echo "|---|---|---|---|---|---|" >> "$LOGF"
# 9B 对称 + 反向
run_test "$M9" "$TMPL9" "v26-9B-t15-t15"  -ngl 99 -c 8192 -ctk turbo1.5 -ctv turbo1.5
run_test "$M9" "$TMPL9" "v26-9B-t15-q8"   -ngl 99 -c 8192 -ctk turbo1.5 -ctv q8_0
# 27B PTQ1_0
run_test "$M27" "$TMPL27" "v26-27B-t15"      -ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo1.5
run_test "$M27" "$TMPL27" "v26-27B-t15-t15"  -ngl 99 -c 4096 -fit off -ctk turbo1.5 -ctv turbo1.5
# 35B MTP（与 35B mtp 段同口径：ncm36 c8192）
run_test "$M35" "$TMPL35" "v26-35B-mtp-t15"  -ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo1.5 --spec-type draft-mtp
echo "=== TURBO1.5 SUPPLEMENT DONE $(date +%H:%M:%S) ===" >> "$LOGF"

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

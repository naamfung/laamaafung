import json, sys, re
rj, slog = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(rj, encoding='utf-8'))
    t = d['timings']
    gps = t['predicted_per_second']
except Exception as e:
    print(f'请求失败: {e}')
    sys.exit(0)
s = open(slog, encoding='utf-8', errors='replace').read()
m3 = re.findall(r'tg_3s = +([0-9.]+)', s)
tg3 = max(float(x) for x in m3) if m3 else ''
print(f'{gps:.1f} (tg_3s峰值 {tg3})')

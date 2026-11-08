import json
from pathlib import Path

configs = ['qwen3.5-4b', 'lfm2.5-vl-3b', 'minicpm-v-4.6', 'specialist_3m', 'specialist_3m_m15']
data = {}
for c in configs:
    p = Path(f'logs/local_eval/mes5_{c}/mes5_evaluation_summary.json')
    if p.exists():
        with open(p, 'r') as f:
            data[c] = json.load(f)

print(f"| {'Class':<20} | {'Qwen Solo':<10} | {'LFM Solo':<10} | {'MiniCPM':<10} | {'3M Base':<10} | {'3M+M15':<10} |")
print("|" + "-"*22 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|")

classes = list(data['specialist_3m_m15']['loose_per_class'].keys())
for cl in classes:
    q = data['qwen3.5-4b']['loose_per_class'].get(cl, 0.0)
    l = data['lfm2.5-vl-3b']['loose_per_class'].get(cl, 0.0)
    m = data['minicpm-v-4.6']['loose_per_class'].get(cl, 0.0)
    e = data['specialist_3m']['loose_per_class'].get(cl, 0.0)
    r = data['specialist_3m_m15']['loose_per_class'].get(cl, 0.0)
    print(f"| {cl:<20} | {q:<10.4f} | {l:<10.4f} | {m:<10.4f} | {e:<10.4f} | {r:<10.4f} |")

print("|" + "-"*22 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|")
print(f"| {'Official Tight-mAP':<20} | {data['qwen3.5-4b']['tight_average_mAP']:^10.4f} | {data['lfm2.5-vl-3b']['tight_average_mAP']:^10.4f} | {data['minicpm-v-4.6']['tight_average_mAP']:^10.4f} | {data['specialist_3m']['tight_average_mAP']:^10.4f} | {data['specialist_3m_m15']['tight_average_mAP']:^10.4f} |")
print(f"| {'Official Loose-mAP':<20} | {data['qwen3.5-4b']['loose_average_mAP']:^10.4f} | {data['lfm2.5-vl-3b']['loose_average_mAP']:^10.4f} | {data['minicpm-v-4.6']['loose_average_mAP']:^10.4f} | {data['specialist_3m']['loose_average_mAP']:^10.4f} | {data['specialist_3m_m15']['loose_average_mAP']:^10.4f} |")
print(f"| {'Active Classes':<20} | {sum(1 for v in data['qwen3.5-4b']['loose_per_class'].values() if v > 0):^10d} | {sum(1 for v in data['lfm2.5-vl-3b']['loose_per_class'].values() if v > 0):^10d} | {sum(1 for v in data['minicpm-v-4.6']['loose_per_class'].values() if v > 0):^10d} | {sum(1 for v in data['specialist_3m']['loose_per_class'].values() if v > 0):^10d} | {sum(1 for v in data['specialist_3m_m15']['loose_per_class'].values() if v > 0):^10d} |")

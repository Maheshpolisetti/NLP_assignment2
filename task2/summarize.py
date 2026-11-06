"""Print mean ± std of test F1 over seeds for each setup in outputs/."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

runs = defaultdict(list)
for f in sorted((Path(__file__).parent / 'outputs').glob('*/results.json')):
    r = json.loads(f.read_text())
    runs[r['run'].rsplit('_seed', 1)[0]].append(r)

keys = ['test_f1_final', 'test_f1_best', 'baseline_right', 'baseline_left', 'baseline_random']
for setup, rs in runs.items():
    print(f"\n{setup}  ({len(rs)} seed(s))")
    for k in keys:
        v = [r[k] for r in rs]
        print(f"  {k:16s}: {np.mean(v):6.2f} ± {np.std(v):.2f}")

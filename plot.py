import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def main(json_path):
    with open(json_path) as f:
        results = json.load(f)
    
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'random': 'gray', 'entropy': 'red', 'info_density': 'blue'}
    x = list(range(1000, 5001, 200))
    
    for name, d in results.items():
        mean = np.array(d['mean'])
        std = np.array(d['std'])
        s = name.split('_')[0]
        ax = a2 if 'SSL' in name else a1
        ax.plot(x, mean, label=s, color=colors[s])
        ax.fill_between(x, mean-std, mean+std, alpha=0.2, color=colors[s])
    
    for ax, t in [(a1, 'No SSL'), (a2, 'With SSL')]:
        ax.set(xlabel='Labeled samples', ylabel='F1-Score', title=t)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs('figures', exist_ok=True)
    basename = os.path.basename(json_path).replace('.json', '.png')
    out_path = os.path.join('figures', basename)
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py results_XXXXXX.json")
        sys.exit(1)
    main(sys.argv[1])

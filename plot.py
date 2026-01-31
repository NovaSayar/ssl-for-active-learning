import json, sys, os
import numpy as np
import matplotlib.pyplot as plt

def main(json_path):
    with open(json_path) as f:
        results = json.load(f)
    
    # First figure: learning curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'random': 'gray', 'entropy': 'red', 'info_density': 'blue'}
    x = list(range(1000, 5001, 200))
    
    for name, d in results.items():
        mean, std = np.array(d['mean']), np.array(d['std'])
        s = name.replace('_base', '').replace('_SSL', '')
        ax = a2 if 'SSL' in name else a1
        ax.plot(x, mean, label=s, color=colors[s])
        ax.fill_between(x, mean-std, mean+std, alpha=0.2, color=colors[s])
    
    for ax, t in [(a1, 'No SSL'), (a2, 'With SSL')]:
        ax.set(xlabel='Labeled samples', ylabel='F1-Score', title=t)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = f"figures/{os.path.basename(json_path).replace('.json', '.png')}"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    
    # Second figure: bar chart of final F1 scores
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    strategies = ['random', 'entropy', 'info_density']
    labels = ['Random', 'Entropy', 'Info Density']
    x_pos = np.arange(len(strategies))
    width = 0.35
    
    base_means = [results[f'{s}_base']['mean'][-1] for s in strategies]
    base_stds = [results[f'{s}_base']['std'][-1] for s in strategies]
    ssl_means = [results[f'{s}_SSL']['mean'][-1] for s in strategies]
    ssl_stds = [results[f'{s}_SSL']['std'][-1] for s in strategies]
    
    ax2.bar(x_pos - width/2, base_means, width, yerr=base_stds, label='No SSL', color='#f4a582', capsize=4)
    ax2.bar(x_pos + width/2, ssl_means, width, yerr=ssl_stds, label='With SSL', color='#92c5de', capsize=4)
    
    ax2.set_ylabel('Final F1-Score')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    out2 = f"figures/{os.path.basename(json_path).replace('.json', '_bars.png')}"
    plt.savefig(out2, dpi=150)
    print(f"Saved: {out2}")
    
    # Third figure: learning curves with log scale
    fig3, (a3, a4) = plt.subplots(1, 2, figsize=(12, 5))
    
    for name, d in results.items():
        mean, std = np.array(d['mean']), np.array(d['std'])
        s = name.replace('_base', '').replace('_SSL', '')
        ax = a4 if 'SSL' in name else a3
        ax.plot(x, mean, label=s, color=colors[s])
        ax.fill_between(x, mean-std, mean+std, alpha=0.2, color=colors[s])
    
    for ax, t in [(a3, 'No SSL (log scale)'), (a4, 'With SSL (log scale)')]:
        ax.set(xlabel='Labeled samples', ylabel='F1-Score', title=t)
        ax.set_yscale('log')
        ax.set_ylim(0.1, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out3 = f"figures/{os.path.basename(json_path).replace('.json', '_log.png')}"
    plt.savefig(out3, dpi=150)
    print(f"Saved: {out3}")

if __name__ == "__main__":
    main(sys.argv[1]) if len(sys.argv) > 1 else print("Usage: python plot.py results.json")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# 1. CONFIG
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 10, 100]
LR_GRID = [0.1, 0.01, 0.001]
WD_GRID = [1e-4, 1e-5]

# 2. MODEL
def make_model(weights_path=None):
    net = models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()
    
    if weights_path and weights_path.exists():
        net.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    
    for p in net.parameters():
        p.requires_grad = False
    net.fc = nn.Linear(512, 10)
    return net.to(DEVICE)

def get_loaders():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
    train = datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    test = datasets.CIFAR10('./data', train=False, download=True, transform=tf)
    return train, DataLoader(test, batch_size=128)

def balanced_init(dataset):
    counts = [0] * 10
    idx = []
    for i, (_, y) in enumerate(dataset):
        if counts[y] < 100:
            idx.append(i)
            counts[y] += 1
        if len(idx) >= 1000:
            break
    return idx

# 3. AL STRATEGIES
def entropy(p):
    return -(p * torch.log(p + 1e-8)).sum(1)

def random_query(idx, n, **kw):
    return [idx[i] for i in np.random.permutation(len(idx))[:n]]

def entropy_query(idx, n, probs, **kw):
    top = entropy(probs).topk(n).indices
    return [idx[i] for i in top]

def density_query(idx, n, probs, sim_cache, **kw):
    top = (entropy(probs) * sim_cache).topk(n).indices
    return [idx[i] for i in top]

STRATEGIES = {'random': random_query, 'entropy': entropy_query, 'info_density': density_query}

# 4. TRAINING
def grid_search_train(model, train_idx, train_data, test_loader):
    n_val = max(1, len(train_idx) // 5)
    val_idx = train_idx[:n_val]
    sub_train_idx = train_idx[n_val:]
    best_f1, best_lr, best_wd = 0, 0.01, 1e-4
    
    for lr in LR_GRID:
        for wd in WD_GRID:
            model.fc = nn.Linear(512, 10).to(DEVICE)
            loader = DataLoader(Subset(train_data, sub_train_idx), batch_size=128, shuffle=True)
            opt = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
            
            model.train()
            for _ in range(10):
                for x, y in loader:
                    opt.zero_grad()
                    nn.CrossEntropyLoss()(model(x.to(DEVICE)), y.to(DEVICE)).backward()
                    opt.step()
            
            model.eval()
            preds, labels = [], []
            val_loader = DataLoader(Subset(train_data, val_idx), batch_size=128)
            with torch.no_grad():
                for x, y in val_loader:
                    preds.append(model(x.to(DEVICE)).argmax(1).cpu())
                    labels.append(y)
            f1 = f1_score(torch.cat(labels), torch.cat(preds), average='macro')
            
            if f1 > best_f1:
                best_f1, best_lr, best_wd = f1, lr, wd
    
    return best_lr, best_wd

def train_eval(model, train_idx, train_data, test_loader, lr=0.01, wd=1e-4):
    model.fc = nn.Linear(512, 10).to(DEVICE)
    loader = DataLoader(Subset(train_data, train_idx), batch_size=128, shuffle=True)
    opt = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    
    model.train()
    for _ in range(10):
        for x, y in loader:
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(x.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step()
    
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(DEVICE)).argmax(1).cpu())
            labels.append(y)
    return f1_score(torch.cat(labels), torch.cat(preds), average='macro')

@torch.no_grad()
def get_probs_feats(model, idx, data):
    loader = DataLoader(Subset(data, idx), batch_size=128)
    fc, model.fc = model.fc, nn.Identity()
    feats = torch.cat([model(x.to(DEVICE)).cpu() for x, _ in loader])
    model.fc = fc
    probs = torch.cat([F.softmax(model(x.to(DEVICE)), 1).cpu() for x, _ in loader])
    return probs, feats

@torch.no_grad()
def compute_sim_cache(feats):
    f = F.normalize(feats, dim=1).half()
    sim = torch.mm(f, f.t()).mean(1)
    return sim.float()

# 5. EXPERIMENT LOOP
def run(strategy, ssl, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_data, test_loader = get_loaders()
    weights = Path("simsiam_resnet18_cifar10_e100.pth") if ssl else None
    model = make_model(weights)
    
    labeled = balanced_init(train_data)
    unlabeled = list(set(range(len(train_data))) - set(labeled))
    fn = STRATEGIES[strategy]
    
    best_lr, best_wd = grid_search_train(model, labeled, train_data, test_loader)
    print(f"  Grid search: lr={best_lr}, wd={best_wd}")
    
    history = []
    for cycle in range(21):
        f1 = train_eval(model, labeled, train_data, test_loader, lr=best_lr, wd=best_wd)
        history.append(f1)
        print(f"  {cycle:2d} | {len(labeled):5d} | {f1:.4f}")
        
        if cycle < 20 and unlabeled:
            probs, feats = get_probs_feats(model, unlabeled, train_data)
            sim_cache = compute_sim_cache(feats)
            new = fn(unlabeled, 200, probs=probs, sim_cache=sim_cache)
            labeled.extend(new)
            unlabeled = [i for i in unlabeled if i not in new]
    
    return history

def main():
    results = {}
    for ssl in [False, True]:
        for s in STRATEGIES:
            name = f"{s}_{'SSL' if ssl else 'base'}"
            print(f"\n=== {name} ===")
            runs = [run(s, ssl, seed) for seed in SEEDS]
            results[name] = {
                'mean': np.mean(runs, 0).tolist(),
                'std': np.std(runs, 0).tolist()
            }
    
    # 6. SAVE RESULTS
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    import json
    with open(f'results_{ts}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    rows = [{
        'exp': n,
        'f1_mean': d['mean'][-1],
        'f1_std': d['std'][-1],
        'auc': np.mean(d['mean'])
    } for n, d in results.items()]
    pd.DataFrame(rows).to_csv(f'results_{ts}.csv', index=False)
    

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    main()

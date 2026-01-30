**Note:** Following changes has been made to have more stable and faster coding process since we use mobile GPUs, also for the reasons mentioned in the methodology.

## Quick Start

```bash
# Install dependencies (Conda)
conda env create -f environment.yml
conda activate ssl_al

# Run the full experiment (6 scenarios, 3 seeds each)
python run_experiment.py

# Or run in background
nohup python -u run_experiment.py > experiment.log 2>&1 &
tail -f experiment.log
```

Output files:
- `results_YYYYMMDD_HHMMSS.json` - Complete data (mean + std per cycle)
- `results_YYYYMMDD_HHMMSS.csv` - Summary table (f1_mean, f1_std, auc)

```bash
# Generate plots from results
python plot.py results_XXXXXX.json
# Output saved to figures/
```

## To ensure the SSL pre-trained weights work correctly with your Active Learning pipeline, follow these mandatory steps:

### 1. Mandatory Model Architecture Changes
The ResNet-18 backbone was optimized for CIFAR-10 (32x32 images). You **must** modify the architecture as follows before loading the weights:

```python
import torch
import torch.nn as nn
from torchvision import models

def get_optimized_resnet18():
    # Load standard ResNet-18
    model = models.resnet18(weights=None)
    
    # Modify for CIFAR-10 small image compatibility
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    # Remove the fully connected layer for feature extraction
    model.fc = nn.Identity()
    return model
```

### 2. Loading the Weights
**a. Download the weights:** You can find the `simsiam_resnet18_cifar10_e100.pth` file under the **Releases** section of this repository.  

**b. Important Transforms:** You must use the following normalization values in your test/query transforms to match the SSL pre-training:

```python
# Use these exact values for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
```

**c. Load it into the optimized backbone:**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = get_optimized_resnet18().to(device)

# Load state dict
state_dict = torch.load("simsiam_resnet18_cifar10_e100.pth", map_location=device)
backbone.load_state_dict(state_dict)

# CRITICAL: Freeze the backbone for Linear Probing (as per Methodology)
for param in backbone.parameters():
    param.requires_grad = False
backbone.eval()
```

### 3. Methodology & Performance Tips
**a. Linear Probing:** We keep the backbone frozen to evaluate the pure representation power of SSL without distortion in the low-data regime.  

**b. Memory Optimization:** Similarity vectors are computed in torch.float16 to reduce memory usage.  

**c. Caching:** The cosine similarity is recomputed each AL cycle on the current unlabeled pool (since the pool shrinks by 200 samples per cycle).

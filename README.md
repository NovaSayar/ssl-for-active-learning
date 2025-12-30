## Active Learning Integration

To use the pre-trained SimSiam backbone for your Active Learning experiments, follow these simple steps:

### 1. Requirements
* Download the backbone: `simsiam_resnet18_cifar10_e100.pth` from the [Releases](../../releases) section.
* Ensure you have the CIFAR-10 dataset in the `/data` folder.

### 2. Loading the Backbone
Use the following logic to initialize your Active Learning model with our SSL weights:

```python
import torch
from train_simsiam import SimSiam # Ensure this is in your path

# Initialize model
model = SimSiam(resnet18) 

# Load the pre-trained weights
checkpoint = torch.load('simsiam_resnet18_cifar10_e100.pth')
model.load_state_dict(checkpoint['state_dict'])

# Extract the backbone for AL
backbone = model.backbone
# Now you can use this backbone for downstream AL tasks!

# If you encounter any path issues, double-check your .gitignore settings.
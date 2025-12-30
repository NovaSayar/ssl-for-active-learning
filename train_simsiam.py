import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from solo.methods import SimSiam

# 1. CIFAR-10 DATA TRANSFORMS & INDEX WRAPPER
class SimSiamTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    def __call__(self, x):
        return self.transform(x), self.transform(x)

class CIFAR10WithIndex(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return index, img, target

# 2. DATASET AND LOADER
train_dataset = CIFAR10WithIndex(root='./data', train=True, download=True, transform=SimSiamTransform())
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

# 3. BACKBONE PREPARATION
backbone = models.resnet18(weights=None)
backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
backbone.maxpool = nn.Identity()
backbone.fc = nn.Identity()

# 4. SIMSIAM INITIALIZATION
model = SimSiam(
    backbone=backbone,
    encoder="resnet18",
    num_classes=10,
    backbone_args={},
    max_epochs=100,
    batch_size=128,
    optimizer="sgd",
    lars=False,
    classifier_lr=0.1,
    exclude_bias_n_norm=False,
    accumulate_grad_batches=1,
    extra_optimizer_args={"momentum": 0.9},
    weight_decay=1e-4,
    scheduler="warmup_cosine",
    min_lr=0.0,
    warmup_start_lr=0.0,
    warmup_epochs=10,
    num_large_crops=2,
    num_small_crops=0,
    proj_hidden_dim=2048,
    proj_output_dim=2048,
    pred_hidden_dim=512,
    lr=0.03
)

# 5. TRAINING CONFIGURATION
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    precision=16,
    log_every_n_steps=10
)

if __name__ == "__main__":
    # Optimize for RTX 4050 Tensor Cores
    torch.set_float32_matmul_precision('medium')
    
    print("Pre-training starting... This will take 100 epochs.")
    trainer.fit(model, train_loader)
    
    save_path = "simsiam_resnet18_cifar10_e100.pth"
    print(f"Training finished. Attempting to save weights...")

    # The 'try-except' block ensures we save the weights even if 
    # the internal attribute names vary between library versions.
    try:
        # Solo-learn typically renames the backbone to 'encoder' internally
        torch.save(model.encoder.state_dict(), save_path) 
        print(f"\n--- SUCCESS: Backbone weights saved at: {save_path} ---")
    except AttributeError:
        # Fallback if the name remained 'backbone'
        torch.save(model.backbone.state_dict(), save_path)
        print(f"\n--- SUCCESS: Backbone weights saved at: {save_path} ---")
    except Exception as e:
        # Emergency backup of the entire model state if something else goes wrong
        torch.save(model.state_dict(), "full_model_backup.pth")
        print(f"Warning: Saved full model backup due to: {e}")

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score, pairwise_distances
import pickle
from typing import List, Tuple, Dict
import random
import pandas as pd
from datetime import datetime

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10Dataset(Dataset):
    """Custom CIFAR-10 Dataset with animal-only filtering"""
    def __init__(self, data_path: str, transform=None, animal_only=True):
        self.data = unpickle(data_path)
        self.images = self.data[b'data']
        self.labels = np.array(self.data[b'labels'])
        self.transform = transform
        
        # Animal categories in CIFAR-10: bird, cat, deer, dog, frog, horse
        self.animal_classes = [2, 3, 4, 5, 6, 7]
        
        if animal_only:
            mask = np.isin(self.labels, self.animal_classes)
            self.images = self.images[mask]
            self.labels = self.labels[mask]
            # Remap labels from 0 to 5
            self.label_map = {old: new for new, old in enumerate(self.animal_classes)}
            self.labels = np.array([self.label_map[l] for l in self.labels])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.images[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def load_pretrained_backbone(checkpoint_path: str, device: str = 'cuda'):
    """
    Load pre-trained weights from solo-learn checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    
    # Remove 'backbone.' prefix
    backbone_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            new_key = key.replace('backbone.', '')
            # Skip projection head and other non-backbone components
            if not any(x in new_key for x in ['projection', 'prototypes', 'momentum']):
                backbone_state_dict[new_key] = value
    
    return backbone_state_dict

def create_cifar_resnet50(num_classes: int = 6, pretrained_path: str = None):
    """
    Create ResNet-50 adapted for CIFAR-10 (32x32 images)
    """
    model = resnet50(pretrained=False)
    
    # Modify first layer for CIFAR-10: 3x3 kernel, stride 1, no maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool
    
    # Load pre-trained weights if available
    if pretrained_path:
        try:
            backbone_weights = load_pretrained_backbone(pretrained_path)
            model.load_state_dict(backbone_weights, strict=False)
            print(f"✓ Pre-trained weights loaded from {pretrained_path}")
        except Exception as e:
            print(f"⚠ Unable to load weights: {e}")
    
    # Replace final classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

class ActiveLearningPipeline:
    """
    Active Learning Pipeline for CIFAR-10 with multiple sampling strategies
    """
    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module,
        device: str = 'cuda',
        initial_size: int = 100,
        query_size: int = 50,
        num_classes: int = 6,
        sampling_strategy: str = 'entropy'
    ):
        self.dataset = dataset
        self.model = model.to(device)
        self.device = device
        self.query_size = query_size
        self.num_classes = num_classes
        self.sampling_strategy = sampling_strategy
        
        # Labeled and unlabeled indices
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        self.labeled_indices = all_indices[:initial_size]
        self.unlabeled_indices = all_indices[initial_size:]
        
        print(f"Initialized with {len(self.labeled_indices)} labeled samples")
        print(f"Unlabeled pool: {len(self.unlabeled_indices)} samples")
        print(f"Sampling strategy: {sampling_strategy}")
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train model for one epoch"""
        self.model.train()
        total_loss = 0
        predictions, targets = [], []
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(outputs.argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(targets, predictions)
        return total_loss / len(dataloader), accuracy
    
    def evaluate(self, dataloader):
        """Evaluate model on test set"""
        self.model.eval()
        predictions, targets = [], []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                predictions.extend(outputs.argmax(1).cpu().numpy())
                targets.extend(labels.numpy())
        
        return accuracy_score(targets, predictions)
    
    def get_embeddings(self, dataloader):
        """Extract feature embeddings from the model (before final FC layer)"""
        self.model.eval()
        embeddings = []
        
        # Temporarily remove the final FC layer
        original_fc = self.model.fc
        self.model.fc = nn.Identity()
        
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                feats = self.model(images)
                embeddings.append(feats.cpu().numpy())
        
        # Restore the FC layer
        self.model.fc = original_fc
        
        return np.vstack(embeddings)
    
    def random_sampling(self, dataloader) -> List[int]:
        """
        Baseline: Random sampling strategy
        """
        # Randomly select query_size samples from unlabeled pool
        selected_indices = np.random.choice(
            len(self.unlabeled_indices), 
            size=min(self.query_size, len(self.unlabeled_indices)), 
            replace=False
        )
        return [self.unlabeled_indices[i] for i in selected_indices]
    
    def entropy_sampling(self, dataloader) -> List[int]:
        """
        Uncertainty sampling: Entropy-based selection
        Selects samples with highest prediction entropy
        """
        self.model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                # Calculate entropy: -sum(p * log(p))
                entropy = -(probs * torch.log(probs + 1e-10)).sum(1)
                uncertainties.extend(entropy.cpu().numpy())
        
        # Select top-k most uncertain samples
        uncertain_indices = np.argsort(uncertainties)[-self.query_size:]
        return [self.unlabeled_indices[i] for i in uncertain_indices]
    
    def least_confidence_sampling(self, dataloader) -> List[int]:
        """
        Uncertainty sampling: Least confidence
        Selects samples where the model is least confident (lowest max probability)
        """
        self.model.eval()
        confidences = []
        
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                # Get maximum probability (confidence)
                max_probs = probs.max(1)[0]
                confidences.extend(max_probs.cpu().numpy())
        
        # Select samples with lowest confidence
        uncertain_indices = np.argsort(confidences)[:self.query_size]
        return [self.unlabeled_indices[i] for i in uncertain_indices]
    
    def margin_sampling(self, dataloader) -> List[int]:
        """
        Uncertainty sampling: Margin sampling
        Selects samples with smallest margin between top two predictions
        """
        self.model.eval()
        margins = []
        
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                # Get top 2 probabilities
                top2 = torch.topk(probs, 2, dim=1)[0]
                # Calculate margin (difference between top 2)
                margin = top2[:, 0] - top2[:, 1]
                margins.extend(margin.cpu().numpy())
        
        # Select samples with smallest margins
        uncertain_indices = np.argsort(margins)[:self.query_size]
        return [self.unlabeled_indices[i] for i in uncertain_indices]
    
    def coreset_sampling(self, dataloader) -> List[int]:
        """
        Diversity-based: Core-set selection
        Selects samples that are most representative of the unlabeled pool
        Uses greedy k-center algorithm on feature embeddings
        """
        # Get embeddings for unlabeled samples
        unlabeled_embeddings = self.get_embeddings(dataloader)
        
        # Get embeddings for labeled samples
        labeled_dataset = Subset(self.dataset, self.labeled_indices)
        labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=False)
        labeled_embeddings = self.get_embeddings(labeled_loader)
        
        # Greedy k-center selection
        selected_indices = []
        
        for _ in range(min(self.query_size, len(self.unlabeled_indices))):
            if len(selected_indices) == 0:
                # Calculate distances to all labeled samples
                distances = pairwise_distances(unlabeled_embeddings, labeled_embeddings)
                min_distances = distances.min(axis=1)
            else:
                # Calculate distances to labeled + already selected samples
                selected_embeddings = unlabeled_embeddings[selected_indices]
                all_labeled = np.vstack([labeled_embeddings, selected_embeddings])
                distances = pairwise_distances(unlabeled_embeddings, all_labeled)
                min_distances = distances.min(axis=1)
            
            # Select sample with maximum minimum distance (farthest from labeled set)
            max_idx = min_distances.argmax()
            selected_indices.append(max_idx)
            min_distances[max_idx] = -1  # Mark as selected
        
        return [self.unlabeled_indices[i] for i in selected_indices]
    
    def query_samples(self) -> List[int]:
        """
        Execute query on unlabeled pool using the specified sampling strategy
        """
        unlabeled_dataset = Subset(self.dataset, self.unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=False)
        
        # Select sampling strategy
        if self.sampling_strategy == 'random':
            queried_indices = self.random_sampling(unlabeled_loader)
        elif self.sampling_strategy == 'entropy':
            queried_indices = self.entropy_sampling(unlabeled_loader)
        elif self.sampling_strategy == 'least_confidence':
            queried_indices = self.least_confidence_sampling(unlabeled_loader)
        elif self.sampling_strategy == 'margin':
            queried_indices = self.margin_sampling(unlabeled_loader)
        elif self.sampling_strategy == 'coreset':
            queried_indices = self.coreset_sampling(unlabeled_loader)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Update labeled/unlabeled pools
        self.labeled_indices.extend(queried_indices)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in queried_indices]
        
        return queried_indices
    
    def active_learning_loop(self, num_rounds: int = 10, epochs_per_round: int = 20):
        """
        Main Active Learning loop
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        results = {'rounds': [], 'train_acc': [], 'test_acc': [], 'labeled_size': []}
        
        # Test dataset (fixed)
        test_dataset = CIFAR10Dataset('/home/seb/ssl-il/cifar-10-batches-py/test_batch', transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        for round_num in range(num_rounds):
            print(f"\n{'='*50}")
            print(f"Round {round_num + 1}/{num_rounds}")
            print(f"Labeled samples: {len(self.labeled_indices)}")
            
            # Create dataloader for labeled data
            labeled_dataset = Subset(self.dataset, self.labeled_indices)
            train_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)
            
            # Training
            optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs_per_round):
                loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
                
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{epochs_per_round} - Loss: {loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Evaluation
            test_acc = self.evaluate(test_loader)
            print(f"✓ Test Accuracy: {test_acc:.4f}")
            
            results['rounds'].append(round_num + 1)
            results['train_acc'].append(train_acc)
            results['test_acc'].append(test_acc)
            results['labeled_size'].append(len(self.labeled_indices))
            
            # Query new samples
            if round_num < num_rounds - 1:
                queried = self.query_samples()
                print(f"→ Queried {len(queried)} new samples")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results

def plot_results(all_results: Dict[str, dict], output_name: str = 'active_learning_comparison.png'):
    """
    Visualize Active Learning results for multiple strategies
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    markers = ['o', 's', 'D', '^', 'v']
    
    # Plot 1: Test Accuracy vs Rounds
    for idx, (strategy, results) in enumerate(all_results.items()):
        ax1.plot(results['rounds'], results['test_acc'], 
                marker=markers[idx % len(markers)], 
                color=colors[idx % len(colors)],
                label=strategy, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Active Learning Round', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy vs AL Rounds', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy vs Dataset Size
    for idx, (strategy, results) in enumerate(all_results.items()):
        ax2.plot(results['labeled_size'], results['test_acc'], 
                marker=markers[idx % len(markers)], 
                color=colors[idx % len(colors)],
                label=strategy, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Labeled Dataset Size', fontsize=12)
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.set_title('Test Accuracy vs Labeled Data', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results plot saved to '{output_name}'")

def save_results_to_csv(all_results: Dict[str, dict], output_name: str = 'active_learning_results.csv'):
    """
    Save Active Learning results to CSV file
    """
    rows = []
    for strategy, results in all_results.items():
        for i in range(len(results['rounds'])):
            rows.append({
                'Strategy': strategy,
                'Round': results['rounds'][i],
                'Labeled_Size': results['labeled_size'][i],
                'Train_Accuracy': results['train_acc'][i],
                'Test_Accuracy': results['test_acc'][i]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_name, index=False)
    print(f"✓ Results saved to '{output_name}'")

# ============ MAIN ============
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"{'='*70}")
    print("ACTIVE LEARNING COMPARISON ON CIFAR-10 (ANIMAL CLASSES)")
    print(f"{'='*70}\n")
    
    # Setup dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = CIFAR10Dataset(
        '/home/seb/ssl-il/cifar-10-batches-py/data_batch_1',
        transform=transform,
        animal_only=True
    )
    
    # Define sampling strategies to compare
    strategies = ['random', 'entropy', 'least_confidence', 'margin', 'coreset']
    all_results = {}
    
    # Run Active Learning for each strategy
    for strategy in strategies:
        print(f"\n{'#'*70}")
        print(f"# RUNNING STRATEGY: {strategy.upper()}")
        print(f"{'#'*70}")
        
        # Create fresh model for each strategy
        model = create_cifar_resnet50(
            num_classes=6,
            pretrained_path=None  # Replace with path to solo-learn checkpoint if available
        )
        
        # Initialize Active Learning pipeline
        al_pipeline = ActiveLearningPipeline(
            dataset=dataset,
            model=model,
            device=device,
            initial_size=100,
            query_size=50,
            num_classes=6,
            sampling_strategy=strategy
        )
        
        # Run Active Learning loop
        results = al_pipeline.active_learning_loop(
            num_rounds=10,
            epochs_per_round=20
        )
        
        all_results[strategy] = results
    
    # Save and visualize all results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_results(all_results, f'al_comparison_{timestamp}.png')
    save_results_to_csv(all_results, f'al_results_{timestamp}.csv')
    
    # Print summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    for strategy, results in all_results.items():
        final_acc = results['test_acc'][-1]
        final_size = results['labeled_size'][-1]
        print(f"{strategy.upper():20s} | Final Acc: {final_acc:.4f} | Labeled: {final_size}")
    print(f"{'='*70}\n")
# Active Learning for CIFAR-10 Animal Classification

This project implements and compares multiple **Active Learning** sampling strategies on CIFAR-10 dataset, focusing on animal classes only.

## 📋 Overview

Active Learning is a semi-supervised machine learning approach where the model iteratively selects the most informative samples to be labeled, maximizing performance while minimizing labeling cost.

### What This Code Does

1. **Trains a ResNet-50 classifier** on CIFAR-10 animal classes (6 classes: bird, cat, deer, dog, frog, horse)
2. **Compares 5 different sampling strategies** to select which samples to label next
3. **Tracks performance** across 10 rounds of Active Learning
4. **Generates comparison plots and CSV results**

---

## 🎯 Sampling Strategies Implemented

### 1. **Random Sampling** (Baseline)
- **Description**: Randomly selects samples from the unlabeled pool
- **Use Case**: Baseline comparison to measure improvement of other strategies
- **Pros**: Simple, unbiased
- **Cons**: Ignores model uncertainty and data distribution

### 2. **Entropy-based Uncertainty Sampling**
- **Description**: Selects samples with highest prediction entropy: `-Σ p(y) log p(y)`
- **Use Case**: When model confusion indicates informativeness
- **Pros**: Captures overall uncertainty across all classes
- **Cons**: May oversample outliers or noisy data

### 3. **Least Confidence Sampling**
- **Description**: Selects samples where the model has lowest confidence (smallest max probability)
- **Use Case**: When you want to query samples the model is most uncertain about
- **Pros**: Simple, intuitive measure of uncertainty
- **Cons**: Only considers top prediction, ignores other classes

### 4. **Margin Sampling**
- **Description**: Selects samples with smallest margin between top-2 predictions
- **Use Case**: When decision boundary is important (binary-like scenarios)
- **Pros**: Focuses on ambiguous cases between top classes
- **Cons**: Ignores information from lower-ranked classes

### 5. **Core-Set Sampling** (Diversity-based)
- **Description**: Uses greedy k-center algorithm on feature embeddings to select diverse, representative samples
- **Use Case**: When you want coverage of the entire data distribution
- **Pros**: Ensures diversity, avoids redundant samples
- **Cons**: Computationally expensive, requires feature extraction

---

## 🏗️ Architecture Details

### ResNet-50 Adaptation for CIFAR-10
Standard ResNet-50 is designed for ImageNet (224×224 images). We adapt it for CIFAR-10 (32×32):

```python
# Modifications:
- Conv1: 7×7 kernel → 3×3 kernel (stride=1)
- Removed MaxPool layer (would lose too much spatial information)
- Final FC layer: 1000 classes → 6 classes (animal-only)
```

### Solo-Learn Integration
The code includes a **bridge function** to load self-supervised pre-trained weights from [solo-learn](https://github.com/vturrisi/solo-learn):

```python
load_pretrained_backbone(checkpoint_path)
```

**What it handles:**
- Strips `backbone.` prefix from checkpoint keys
- Filters out projection head layers (used only during SSL pre-training)
- Loads only the backbone encoder weights

**To use pre-trained weights:**
```python
model = create_cifar_resnet50(
    num_classes=6,
    pretrained_path='/path/to/solo-learn-checkpoint.ckpt'
)
```

---

## 🚀 Usage

### Requirements
```bash
pip install torch torchvision scikit-learn matplotlib pandas numpy
```

### Run Comparison
```bash
python3 main.py
```

This will:
1. Run all 5 sampling strategies sequentially
2. Train for 10 rounds with 20 epochs per round
3. Save results to:
   - `al_results_TIMESTAMP.csv` - Detailed metrics
   - `al_comparison_TIMESTAMP.png` - Visualization plots

---

## 📊 Output Files

### CSV Structure
```
Strategy,Round,Labeled_Size,Train_Accuracy,Test_Accuracy
random,1,100,0.45,0.42
entropy,1,100,0.48,0.45
...
```

### Plots Generated
1. **Test Accuracy vs Round**: Shows learning progress over AL rounds
2. **Test Accuracy vs Dataset Size**: Shows sample efficiency

---

## 🔧 Configuration

Modify these parameters in `main.py`:

```python
# Active Learning settings
initial_size = 100      # Initial labeled samples
query_size = 50         # Samples to label per round
num_rounds = 10         # Number of AL iterations
epochs_per_round = 20   # Training epochs per round

# Model settings
num_classes = 6         # Animal classes only
device = 'cuda'         # Use GPU if available
```

---

## 📖 Key Concepts

### Active Learning Loop
```
1. Start with small labeled set (100 samples)
2. Train model on labeled data
3. Use model to score unlabeled samples
4. Query most informative samples (50 samples)
5. Add queried samples to labeled set
6. Repeat steps 2-5 for N rounds
```

### Why Animal Classes Only?
- Reduces problem complexity for experimentation
- 6 classes: bird (2), cat (3), deer (4), dog (5), frog (6), horse (7)
- Labels are remapped to 0-5 for model training
- Test set also filtered to animal classes only

### Memory Management
- GPU memory is cleared after each round: `torch.cuda.empty_cache()`
- Prevents OOM errors during long training runs

---

## 🎓 When to Use Each Strategy

| Strategy | Best For |
|----------|----------|
| **Random** | Baseline comparison |
| **Entropy** | Multi-class problems with balanced confusion |
| **Least Confidence** | When top prediction matters most |
| **Margin** | Binary or near-binary decision tasks |
| **Core-Set** | Diverse data coverage, avoiding redundancy |

---

## 🔬 Experimental Notes

- **Data Simulation**: CIFAR-10 labels are known, but we simulate "unlabeled" data
- **Fair Comparison**: Each strategy starts with identical model initialization
- **Evaluation**: Fixed test set ensures consistent evaluation across strategies
- **Reproducibility**: Random seed should be set for deterministic results

---

## 🐛 Troubleshooting

### Matplotlib Backend Error
If you see `AttributeError: module '_tkinter'`, the code already handles this:
```python
matplotlib.use('Agg')  # Non-interactive backend
```

### CUDA Out of Memory
Reduce batch size or model complexity:
```python
train_loader = DataLoader(labeled_dataset, batch_size=16, ...)  # Default: 32
```

### Slow Core-Set Sampling
Core-set requires pairwise distance computation. For large unlabeled pools:
- Use approximate nearest neighbor algorithms
- Subsample unlabeled pool before core-set selection

---

## 📚 References

1. **Active Learning Literature**: Settles, B. (2009). Active Learning Literature Survey
2. **Solo-Learn**: Self-Supervised Learning library - [GitHub](https://github.com/vturrisi/solo-learn)
3. **Core-Set**: Sener & Savarese (2018). "Active Learning for Convolutional Neural Networks"

---

## 📝 License

This code is provided for research and educational purposes.

---

## 🤝 Contributing

To add new sampling strategies:
1. Implement a new method in `ActiveLearningPipeline` class
2. Add strategy name to `strategies` list in main
3. Update README with strategy description

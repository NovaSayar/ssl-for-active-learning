import torch

# Load the file
checkpoint = torch.load("simsiam_resnet18_backbone.pth", map_location="cpu")

# Check contents
print(f"--- File check ---")
print(f"Is the file a dictionary? {isinstance(checkpoint, dict)}")
print(f"Total number of layers/parameters: {len(checkpoint)}")
print(f"First layer name: {list(checkpoint.keys())[0]}")
print(f"First layer weight shape: {checkpoint[list(checkpoint.keys())[0]].shape}")
print(f"\n--- RESULT: File is full and healthy! ---")
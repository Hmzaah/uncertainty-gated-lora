# 🛠 Adapter Configuration Schema

This folder contains the blueprints for our experts.

## Structure (`.yaml`)
```yaml
method: "lora"       # Options: lora, dora, pissa
backbone: "resnet18" # Options: resnet18, resnet50
hyperparameters:
  r: 16              # Rank (Efficiency vs Capacity)
  alpha: 32          # Scaling factor (usually 2x rank)
  dropout: 0.1       # Regularization
  target_modules:    # For ResNet, we target conv layers
    - "layer4.0.conv1"
    - "layer4.0.conv2"

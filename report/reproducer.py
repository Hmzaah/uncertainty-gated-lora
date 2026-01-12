import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "microsoft/resnet-50"
processor = AutoImageProcessor.from_pretrained(model_id)
model = ResNetForImageClassification.from_pretrained(model_id).to(device)
model.eval()

DOMAINS = ["test_sunny", "test_rain", "test_night"]
COLORS = ['#2ecc71', '#e74c3c', '#e67e22'] # Green, Red, Orange

def get_avg_entropy(folder_name):
    path = f"data/{folder_name}"
    if not os.path.exists(path): return 0.0
    
    entropies = []
    for img_name in os.listdir(path):
        if img_name.lower().endswith(('.jpg', '.png')):
            try:
                img = Image.open(os.path.join(path, img_name)).convert("RGB")
                inputs = processor(img, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1)
                ent = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).item()
                entropies.append(ent)
            except Exception as e:
                print(f"Skipping corrupt image: {img_name}")
            
    return np.mean(entropies) if entropies else 0.0


def visualize():
    print("Generating Entropy Report Graph...")
    results = [get_avg_entropy(d) for d in DOMAINS]
    labels = ["Sunny\n(Baseline)", "Rain\n(OOD)", "Night\n(OOD)"]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, results, color=COLORS, alpha=0.8, edgecolor='black')
    
    # Add Threshold Line
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Uncertainty Threshold (1.0)')
    
    plt.ylabel("Shannon Entropy (Uncertainty)")
    plt.title("ResNet-50 Failure Analysis: Domain Shift")
    plt.legend()
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("report/entropy_plot.png", dpi=300)
    print("✅ Graph saved to report/entropy_plot.png")

if __name__ == "__main__":
    visualize()

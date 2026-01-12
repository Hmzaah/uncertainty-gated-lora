import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm

# Add root to path so we can import 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.factory import AdapterFactory
from core.dataset import RobustDataset

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data folder")
    args = parser.parse_args()

    # 1. Build Model via Factory
    model, cfg = AdapterFactory.create_model(args.config)
    device = cfg['training']['device'] if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 2. Prepare Data
    print(f"📂 [Trainer] Loading data from: {args.data_dir}")
    train_dataset = RobustDataset(args.data_dir, is_training=True)
    
    if len(train_dataset) == 0:
        print("❌ Error: No images found. Check your folder structure.")
        return

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True
    )
    
    print(f"📊 [Trainer] Batches per epoch: {len(train_loader)}")
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()


    # 4. Training Loop
    model.train()
    print("🚀 [Trainer] Starting Training Loop...")
    
    for epoch in range(cfg['training']['epochs']):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(train_loader)
        print(f"   ✅ Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")

    # 5. Save Adapter
    save_path = os.path.join(cfg['project']['output_dir'], "adapter_final")
    print(f"💾 [Trainer] Saving adapter to: {save_path}")
    model.save_pretrained(save_path)
    print("✨ Training Complete.")

if __name__ == "__main__":
    train()

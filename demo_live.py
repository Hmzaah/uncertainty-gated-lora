import os
import glob
import time
import random
import argparse
import sys

# Try to import Phase 4 brain (Handle missing module gracefully)
try:
    from core.inference import DynamicEdgeSystem
except ImportError:
    DynamicEdgeSystem = None

def auto_detect_bdd_path(root="data"):
    """
    Auto-detects BDD100K images. 'Pick and Place' logic.
    """
    print(f"🔍 [System] Scanning '{root}' for BDD100K images...")
    
    # Prioritize deep paths first
    patterns = [
        "bdd100k/images/100k/train",
        "bdd100k/images/10k/train",
        "images/100k/train",
        "**/train"
    ]
    
    for pat in patterns:
        candidates = glob.glob(os.path.join(root, pat), recursive=True)
        for path in candidates:
            # Verify it actually has images
            if glob.glob(os.path.join(path, "*.jpg")):
                print(f"✅ [System] Auto-detected Dataset at: {path}")
                return path

    print("❌ [System] Could not find BDD100K images. Please place them in 'data/'")
    return None

def run_simulation(image_dir):
    if DynamicEdgeSystem is None:
        print("\n⚠️  [System] Core Inference engine not found. Phase 4 pending.")
        return

    system = DynamicEdgeSystem(base_model_name="microsoft/resnet-50")
    
    # Load available experts
    experts = {
        "Sunny-Expert": "checkpoints/sunny/adapter_final",
        "Rain-Expert": "checkpoints/rain/adapter_final",
        "Night-DoRA": "checkpoints/night/adapter_final"
    }
    
    for name, path in experts.items():
        if os.path.exists(path):
            system.load_adapter(path, name)

    print(f"\n🚗 [Demo] Starting Simulation using images from: {image_dir}")
    images = random.sample(glob.glob(os.path.join(image_dir, "*.jpg")), 5)
    
    for img_path in images:
        result = system.predict(img_path)
        print(f"🖼️ {os.path.basename(img_path)} -> {result['prediction']} ({result['confidence']:.2f}) | {result['expert']}")
        time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data", help="Root folder to scan")
    args = parser.parse_args()
    
    # Auto-detect logic
    img_path = auto_detect_bdd_path(args.data_root)
    
    if img_path:
        run_simulation(img_path)

import json
import os
import shutil
from tqdm import tqdm
import argparse

def sort_custom_bdd(label_path, image_dir, output_dir, limit=2000):
    """
    Sorts BDD100K images from a specific folder into Weather -> Scene hierarchy.
    """
    # 1. Verification
    if not os.path.exists(label_path):
        print(f"❌ Error: Label file missing: {label_path}")
        return
    if not os.path.exists(image_dir):
        print(f"❌ Error: Image directory missing: {image_dir}")
        return

    # 2. Setup Mappings
    SCENE_MAPPING = {'city street': 'city', 'highway': 'highway', 'residential': 'residential'}
    
    def get_domain(weather, timeofday):
        if weather == 'rainy': return 'rain'
        if timeofday == 'night': return 'night'
        if weather == 'clear' and timeofday == 'daytime': return 'sunny'
        return None

    print(f"📖 Reading labels...")
    with open(label_path, 'r') as f:
        labels = json.load(f)

    print(f"🚀 Sorting images from {image_dir}...")
    counts = {d: {s: 0 for s in SCENE_MAPPING.values()} for d in ['sunny', 'rain', 'night']}
    
    # Pre-check existing images to save time
    available_images = set(os.listdir(image_dir))
    
    for entry in tqdm(labels):
        img_name = entry['name']
        attrs = entry['attributes']
        
        # A. Check if image exists
        if img_name not in available_images:
            continue
        
        # B. Check Domain/Class
        domain = get_domain(attrs['weather'], attrs['timeofday'])
        if not domain: continue
        
        scene = attrs['scene']
        if scene not in SCENE_MAPPING: continue
        target_class = SCENE_MAPPING[scene]

        if counts[domain][target_class] >= limit: continue

        # C. Copy File
        src_path = os.path.join(image_dir, img_name)
        dest_folder = os.path.join(output_dir, domain, target_class)
        os.makedirs(dest_folder, exist_ok=True)
        
        shutil.copy(src_path, os.path.join(dest_folder, img_name))
        counts[domain][target_class] += 1
    
    print("\n✅ Sorting Complete.")
    for d in counts:
        print(f"--- {d.upper()} ---")
        print(counts[d])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to JSON labels")
    parser.add_argument("--images", required=True, help="Path to folder containing .jpg images")
    parser.add_argument("--dest", default="./data/bdd_scenes", help="Output folder")
    parser.add_argument("--limit", type=int, default=2000, help="Max images per class")
    args = parser.parse_args()
    
    sort_custom_bdd(args.labels, args.images, args.dest, args.limit)

import json
import os
import shutil
from tqdm import tqdm
import argparse

def sort_bdd_scenes(source_dir, json_path, output_dir, limit=2000):
    """
    Sorts BDD100K images into a Weather -> Scene hierarchy.
    Output Structure:
       data/bdd_scenes/sunny/highway/img001.jpg
    """
    
    # 1. Define the Tasks (Classes)
    SCENE_MAPPING = {
        'city street': 'city',
        'highway': 'highway', 
        'residential': 'residential'
    }
    
    # 2. Define the Domains (Experts)
    def get_domain(weather, timeofday):
        if weather == 'rainy': return 'rain'
        if timeofday == 'night': return 'night'
        if weather == 'clear' and timeofday == 'daytime': return 'sunny'
        return None

    print(f"📖 Reading labels from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            labels = json.load(f)
    except FileNotFoundError:
        print("❌ Error: Label file not found. Please check the path.")
        return

    print(f"🚀 Sorting images into {output_dir}...")
    
    # Counter to ensure balanced data
    counts = {d: {s: 0 for s in SCENE_MAPPING.values()} for d in ['sunny', 'rain', 'night']}
    
    for entry in tqdm(labels):
        img_name = entry['name']
        attrs = entry['attributes']
        
        # A. Filter by Domain
        domain = get_domain(attrs['weather'], attrs['timeofday'])
        if not domain: continue
        
        # B. Filter by Scene
        raw_scene = attrs['scene']
        if raw_scene not in SCENE_MAPPING: continue
        target_class = SCENE_MAPPING[raw_scene]
        
        # C. Check Limit
        if counts[domain][target_class] >= limit:
            continue
            
        # D. Copy File
        src_path = os.path.join(source_dir, img_name)
        if not os.path.exists(src_path):
            # Try appending .jpg if missing
            if os.path.exists(src_path + ".jpg"): src_path += ".jpg"
            else: continue
            
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
    parser.add_argument("--images", required=True, help="Folder containing raw 10k images")
    parser.add_argument("--labels", required=True, help="Path to json label file")
    parser.add_argument("--dest", default="./data/bdd_scenes", help="Where to save the sorted dataset")
    parser.add_argument("--limit", type=int, default=2000, help="Max images per class")
    args = parser.parse_args()
    
    sort_bdd_scenes(args.images, args.labels, args.dest, args.limit)

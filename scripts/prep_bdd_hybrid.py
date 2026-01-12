import json
import os
import shutil
from tqdm import tqdm
import argparse

def sort_hybrid(label_path, image_root, output_dir, limit=2000):
    """
    Matches OFFICIAL BDD labels (Attributes) with DATASETNINJA images (Pixels).
    """
    SCENE_MAPPING = {'city street': 'city', 'highway': 'highway', 'residential': 'residential'}
    
    def get_domain(weather, timeofday):
        if weather == 'rainy': return 'rain'
        if timeofday == 'night': return 'night'
        if weather == 'clear' and timeofday == 'daytime': return 'sunny'
        return None

    print(f"📖 Loading Official Labels from {label_path}...")
    with open(label_path, 'r') as f:
        labels = json.load(f)

    print(f"🔍 Indexing your images in {image_root} (this takes a moment)...")
    # Build a dictionary: "img123.jpg" -> "/path/to/img123.jpg"
    available_images = {}
    for root, dirs, files in os.walk(image_root):
        for f in files:
            if f.endswith('.jpg'):
                available_images[f] = os.path.join(root, f)

    print(f"   ✅ Found {len(available_images)} images available.")
    print(f"🚀 Sorting into {output_dir}...")
    
    counts = {d: {s: 0 for s in SCENE_MAPPING.values()} for d in ['sunny', 'rain', 'night']}
    
    for entry in tqdm(labels):
        img_name = entry['name']
        attrs = entry['attributes']
        
        # 1. Do we have this image?
        if img_name not in available_images:
            continue
            
        # 2. Check Domain (Weather)
        domain = get_domain(attrs['weather'], attrs['timeofday'])
        if not domain: continue
        
        # 3. Check Scene
        scene = attrs['scene']
        if scene not in SCENE_MAPPING: continue
        target_class = SCENE_MAPPING[scene]
        
        # 4. Check Limit
        if counts[domain][target_class] >= limit: continue
        
        # 5. Copy
        src = available_images[img_name]
        dest_folder = os.path.join(output_dir, domain, target_class)
        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy(src, os.path.join(dest_folder, img_name))
        
        counts[domain][target_class] += 1

    print("\n✅ Hybrid Sorting Complete.")
    for d in counts:
        print(f"--- {d.upper()} ---")
        print(counts[d])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--dest", default="./data/bdd_scenes")
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    
    sort_hybrid(args.labels, args.images, args.dest, args.limit)

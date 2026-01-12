import json
import os
import shutil
from tqdm import tqdm
import argparse

def sort_ninja_scenes(source_root, output_dir, limit=2000):
    """
    Sorts DatasetNinja formatted BDD100K (split JSONs) into Weather -> Scene hierarchy.
    """
    
    # 1. Define Mappings
    SCENE_MAPPING = {
        'city street': 'city',
        'highway': 'highway', 
        'residential': 'residential'
    }
    
    def get_domain(weather, timeofday):
        if weather == 'rainy': return 'rain'
        if timeofday == 'night': return 'night'
        if weather == 'clear' and timeofday == 'daytime': return 'sunny'
        return None

    counts = {d: {s: 0 for s in SCENE_MAPPING.values()} for d in ['sunny', 'rain', 'night']}

    # 2. Iterate through both 'train' and 'val' folders to get enough data
    for split in ['train', 'val']:
        split_root = os.path.join(source_root, split)
        ann_dir = os.path.join(split_root, 'ann')
        img_dir = os.path.join(split_root, 'img')

        if not os.path.exists(ann_dir):
            continue

        print(f"🚀 Processing {split} set from {ann_dir}...")
        files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]

        for ann_file in tqdm(files):
            try:
                with open(os.path.join(ann_dir, ann_file), 'r') as f:
                    data = json.load(f)
            except:
                continue

            attrs = data.get('attributes') or data
            weather = attrs.get('weather')
            timeofday = attrs.get('timeofday')
            scene = attrs.get('scene')

            if not weather or not timeofday or not scene: continue

            # A. Check Domain/Class
            domain = get_domain(weather, timeofday)
            if not domain: continue

            if scene not in SCENE_MAPPING: continue
            target_class = SCENE_MAPPING[scene]

            if counts[domain][target_class] >= limit: continue

            # B. Find Image (try .jpg match)
            image_name = ann_file.replace('.json', '') 
            src_img_path = os.path.join(img_dir, image_name)

            if not os.path.exists(src_img_path):
                if os.path.exists(src_img_path + ".jpg"): src_img_path += ".jpg"
                else: continue

            # C. Copy File
            dest_folder = os.path.join(output_dir, domain, target_class)
            os.makedirs(dest_folder, exist_ok=True)
            
            shutil.copy(src_img_path, os.path.join(dest_folder, image_name))
            counts[domain][target_class] += 1
    
    print("\n✅ Sorting Complete.")
    for d in counts:
        print(f"--- {d.upper()} ---")
        print(counts[d])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to DatasetNinja root")
    parser.add_argument("--dest", default="./data/bdd_scenes", help="Output folder")
    parser.add_argument("--limit", type=int, default=2000, help="Max images per class")
    args = parser.parse_args()
    
    sort_ninja_scenes(args.source, args.dest, args.limit)

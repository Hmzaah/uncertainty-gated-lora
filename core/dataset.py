import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class RobustDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True):
        """
        Args:
            root_dir (str): Path to the dataset (e.g., 'data/test_sunny').
            transform (callable, optional): Custom transform.
            is_training (bool): If True, applies heavy augmentation.
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        
        # 1. Auto-Detect Classes (Sorted ensures consistency: Class A is always 0)
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"📂 [Dataset] Found Classes: {self.class_to_idx}")

        # 2. Index Images
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_folder, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

        print(f"📊 [Dataset] Loaded {len(self.image_paths)} images from {root_dir}")

        # 3. Define Default Transforms (The Augmentation Strategy)
        if transform:
            self.transform = transform
        else:
            if is_training:
                # Heavy Augmentation for Training (The "Multiplier" Effect)
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # Clean Transform for Validation
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open and convert to RGB (Fixes crash on PNGs with Alpha channel)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

if __name__ == "__main__":
    # Quick Test
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        ds = RobustDataset(path, is_training=True)
        img, lbl = ds[0]
        print(f"✅ Sample Shape: {img.shape} | Label: {lbl}")
    else:
        print("Usage: python core/dataset.py data/test_sunny")

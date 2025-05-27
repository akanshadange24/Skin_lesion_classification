from torch.utils.data import Dataset
from PIL import Image
import os

class CustomMelanomaDataset(Dataset):
    def __init__(self, img_dir=None, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.classes = [d for d in sorted(os.listdir(img_dir)) 
                       if os.path.isdir(os.path.join(img_dir, d)) and not d.startswith('.')]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        self.images = []
        self.labels = []
        for class_name in self.classes:
            class_dir = os.path.join(img_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')) and not img_name.startswith('.'):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

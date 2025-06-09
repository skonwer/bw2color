import os
from torchvision.datasets import Places365
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Paths
output_dir = "places365"
os.makedirs(f"{output_dir}/train", exist_ok=True)
os.makedirs(f"{output_dir}/val", exist_ok=True)

# Transform: Resize to 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset (small=True means 256x256 JPGs)
train = Places365(root=output_dir, split='train-standard', small=True, download=True, transform=transform)
val = Places365(root=output_dir, split='val', small=True, download=True, transform=transform)

train_loader = DataLoader(train, batch_size=1, shuffle=False)
val_loader = DataLoader(val, batch_size=1, shuffle=False)

def dump(loader, split, max_samples=1000):
    for i, (img, _) in enumerate(tqdm(loader, desc=split)):
        if i >= max_samples:
            break
        save_image(img[0], f"{output_dir}/{split}/{i:06d}.jpg")

dump(train_loader, "train", max_samples=1000)
dump(val_loader, "val", max_samples=200)
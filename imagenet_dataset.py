import os
from PIL import Image
import numpy as np

def create_imagenet_like_dataset(root_dir='imagenet', num_classes=5, images_per_class=10, image_size=256):
    splits = ['train', 'val']
    for split in splits:
        split_dir = os.path.join(root_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for class_idx in range(num_classes):
            class_dir = os.path.join(split_dir, f'class{class_idx}')
            os.makedirs(class_dir, exist_ok=True)

            for img_idx in range(images_per_class):
                img = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(class_dir, f'img_{img_idx}.jpg'))

    print(f"Dummy ImageNet-like dataset created at: {root_dir}")

# Example usage
create_imagenet_like_dataset(root_dir='imagenet', num_classes=100, images_per_class=200, image_size=256)
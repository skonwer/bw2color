import os
import torchvision
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def save_cifar100_as_imagenet_style(root_dir="cifar100", img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size))
    ])

    # Download CIFAR-100 train and test datasets
    trainset = torchvision.datasets.CIFAR100(root="data", train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root="data", train=False, download=True)

    for split, dataset in zip(["train", "val"], [trainset, testset]):
        print(f"Processing {split} set...")
        for idx, (img, label) in enumerate(tqdm(dataset)):
            class_name = dataset.classes[label]
            save_dir = os.path.join(root_dir, split, class_name)
            os.makedirs(save_dir, exist_ok=True)

            img_resized = transform(img)
            img_filename = os.path.join(save_dir, f"{idx}.jpg")
            img_resized.save(img_filename)

    print(f"CIFAR-100 data saved to '{root_dir}/train' and '{root_dir}/val'.")

# Run the function
save_cifar100_as_imagenet_style(root_dir="cifar100", img_size=224)
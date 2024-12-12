import os
import random
import shutil

# Define the dataset path and split ratios
base_dir = './data/vipoooool/new-plant-diseases-dataset/versions/2/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'

splits = ["training", "valid", "test"]
split_ratios = {"training": 0.0,"valid": 0.7, "test": 0.3}

# Set random seed for reproducibility
random.seed(42)

def create_split_folders(base_dir, splits, subfolders):
    for split in splits:
        split_path = os.path.join(base_dir, split)
        os.makedirs(split_path, exist_ok=True)
        for subfolder in subfolders:
            os.makedirs(os.path.join(split_path, subfolder), exist_ok=True)

def split_dataset(base_dir, splits, split_ratios):
    # Get the list of subfolders
    subfolders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Create split folders
    create_split_folders(base_dir, splits, subfolders)

    for subfolder in subfolders:
        subfolder_path = os.path.join(base_dir, subfolder)
        images = [f for f in os.listdir(subfolder_path) if f.lower().endswith(".jpg")]
        random.shuffle(images)

        # Calculate split indices
        total_images = len(images)
        train_end = int(total_images * split_ratios["training"])
        val_end = train_end + int(total_images * split_ratios["valid"])

        # Split images
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        # Move images to respective split folders
        for split, split_images in zip(splits, [train_images, val_images, test_images]):
            for image in split_images:
                src_path = os.path.join(subfolder_path, image)
                dest_path = os.path.join(base_dir, split, subfolder, image)
                shutil.copy2(src_path, dest_path)

if __name__ == "__main__":
    split_dataset(base_dir, splits, split_ratios)
    print("Dataset successfully split into training, validation, and testing sets.")

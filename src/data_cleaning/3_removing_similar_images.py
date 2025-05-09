import os
import shutil
from pathlib import Path

import yaml

def load_config():
    with open('src/data_cleaning/data_cleaning_config.yaml', 'r') as file:
        return yaml.safe_load(file)

def copy_unique_images(dataset2_path, dataset3_path, fresh_path):
    # Create fresh directory if it doesn't exist
    Path(fresh_path).mkdir(parents=True, exist_ok=True)

    # Get list of images from both directories
    dataset2_images = set(os.listdir(dataset2_path))
    dataset3_images = set(os.listdir(dataset3_path))

    # Find images that are unique to dataset3 (not in dataset2)
    unique_images = dataset3_images - dataset2_images

    # Copy unique images to fresh folder
    for image in unique_images:
        source_path = os.path.join(dataset3_path, image)
        dest_path = os.path.join(fresh_path, image)
        shutil.copy2(source_path, dest_path)
        print(f"Copied: {image}")

    print(f"\nTotal unique images copied: {len(unique_images)}")

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    cross_dataset_config = config['duplicate_removal']['cross_dataset_comparison']

    # Define paths from config
    dataset2_path = cross_dataset_config['reference_dataset']
    dataset3_path = cross_dataset_config['comparison_dataset']
    fresh_path = cross_dataset_config['destination_folder']

    # Run the function
    copy_unique_images(dataset2_path, dataset3_path, fresh_path)

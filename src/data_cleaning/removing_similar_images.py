import os
import shutil
from pathlib import Path

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
    # Define paths
    dataset2_path = "data/cleaned_data/dedup/dataset2"
    dataset3_path = "data/cleaned_data/dedup/dataset3"
    fresh_path = "data/cleaned_data/dedup/fresh"
    
    # Run the function
    copy_unique_images(dataset2_path, dataset3_path, fresh_path)

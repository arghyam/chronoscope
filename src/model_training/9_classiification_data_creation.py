import os
import shutil
from collections import Counter

# Source and destination directories
src_dir = "dataset/data_cleaned/filtered_data/dataset2"
dest_base_dir = "dataset/data_cleaned/classification_data"
os.makedirs(dest_base_dir, exist_ok=True)

# Initialize counter for class distribution
class_distribution = Counter()

# Create the classification folders
classes = ["Good", "Blurry", "Foggy", "Poor lighting", "Out of focus", "Oriented"]
for class_name in classes:
    # Create folder with class name, replacing spaces with underscores
    class_folder = os.path.join(dest_base_dir, class_name.replace(" ", "_"))
    os.makedirs(class_folder, exist_ok=True)

# Read the annotations file and copy images to respective folders
with open("dataset/data_cleaned/filtered_data/dataset2/annotations.csv", "r") as f:
    # Skip header
    next(f)

    for line in f:
        image_name, annotation, _ = line.strip().split(",")

        # Update distribution counter
        class_distribution[annotation] += 1

        # Source and destination paths
        src_path = os.path.join(src_dir, image_name)
        dest_path = os.path.join(dest_base_dir, annotation.replace(" ", "_"), image_name)

        # Copy the file if it exists
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
        else:
            print(f"Warning: Source file not found: {src_path}")

print("\nImage Classification Distribution:")
print("-" * 40)
total_images = sum(class_distribution.values())

# Print distribution with percentages
for class_name, count in class_distribution.most_common():
    percentage = (count / total_images) * 100
    print(f"{class_name:<15}: {count:>5} images ({percentage:>6.2f}%)")

print("-" * 40)
print(f"Total Images    : {total_images}")

print("Image classification organization completed!")

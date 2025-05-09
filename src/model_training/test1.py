from pathlib import Path

# Define the paths
image_dir = "dataset/training_data/indivisual_numbers/dataset1_1460/images"
label_dir = "dataset/training_data/indivisual_numbers/dataset1_1460/labels"

def find_mismatches():
    # Get list of all files in both directories
    # Handle multiple image extensions
    image_files = set()
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.update(f.stem for f in Path(image_dir).glob(f'*{ext}'))

    label_files = set(f.stem for f in Path(label_dir).glob('*.txt'))

    # Find mismatches
    images_without_labels = image_files - label_files
    labels_without_images = label_files - image_files

    # Print results
    print(f"Total images: {len(image_files)}")
    print(f"Total labels: {len(label_files)}")

    if images_without_labels:
        print("\nImages without corresponding labels:")
        for img in sorted(images_without_labels):
            # Find the actual extension of the image
            for ext in ['.jpg', '.jpeg', '.png']:
                if (Path(image_dir) / f"{img}{ext}").exists():
                    print(f"- {img}{ext}")
                    break

    if labels_without_images:
        print("\nLabels without corresponding images:")
        for label in sorted(labels_without_images):
            print(f"- {label}.txt")

    if not images_without_labels and not labels_without_images:
        print("\nAll files are properly paired!")

if __name__ == "__main__":
    find_mismatches()

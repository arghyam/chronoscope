import hashlib
import shutil
from pathlib import Path

import imagehash
import yaml
from PIL import Image
from tqdm import tqdm

def get_image_hash(image_path, hash_method='phash'):
    """
    Calculate a perceptual hash for an image.

    Args:
        image_path: Path to the image file
        hash_method: Type of hash to use ('phash', 'dhash', 'ahash', or 'md5')

    Returns:
        Hash value as a string
    """
    try:
        if hash_method == 'md5':
            # Calculate MD5 hash of the file content (byte-level comparison)
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        else:
            # Calculate perceptual hash (visual similarity)
            img = Image.open(image_path)
            if hash_method == 'phash':
                return str(imagehash.phash(img))
            elif hash_method == 'dhash':
                return str(imagehash.dhash(img))
            elif hash_method == 'ahash':
                return str(imagehash.average_hash(img))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def remove_duplicate_images(input_folder, output_folder, hash_method='phash'):
    """
    Copy unique images from input folder to output folder, skipping duplicates.

    Args:
        input_folder: Path to the folder containing original images
        output_folder: Path to save unique images
        hash_method: Hash method to use for comparison

    Returns:
        Tuple of (number of unique images, number of duplicates found)
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # Create output directory if it doesn't exist
    output_folder.mkdir(exist_ok=True, parents=True)

    # Load config and get allowed extensions
    with open('src/data_cleaning/data_cleaning_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    allowed_extensions = config['duplicate_removal']['content_based']['allowed_extensions']

    # Get all image files
    image_files = [
        f for f in input_folder.glob('**/*')
        if f.is_file() and f.suffix.lower() in allowed_extensions
    ]

    print(f"Found {len(image_files)} images in {input_folder}")

    # Calculate hashes for all images
    hash_dict = {}
    duplicates = []
    unique_images = []

    for img_path in tqdm(image_files, desc="Calculating image hashes"):
        img_hash = get_image_hash(img_path, hash_method)
        if img_hash:
            if img_hash in hash_dict:
                duplicates.append((img_path, hash_dict[img_hash]))
            else:
                hash_dict[img_hash] = img_path
                unique_images.append(img_path)

    print(f"Found {len(unique_images)} unique images and {len(duplicates)} duplicates")

    # Copy unique images to output folder
    for img_path in tqdm(unique_images, desc="Copying unique images"):
        dest_path = output_folder / img_path.name
        # Handle filename conflicts
        if dest_path.exists():
            base_name = img_path.stem
            extension = img_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = output_folder / f"{base_name}_{counter}{extension}"
                counter += 1
        shutil.copy2(img_path, dest_path)

    return len(unique_images), len(duplicates)

if __name__ == "__main__":
    # Load configuration
    with open('src/data_cleaning/data_cleaning_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Get content-based duplicate removal settings
    content_config = config['duplicate_removal']['content_based']

    # Process each source-destination folder pair
    for source_folder, dest_folder in zip(
        content_config['source_folders'],
        content_config['destination_folders']
    ):
        print(f"\nProcessing folder pair:")
        print(f"Source: {source_folder}")
        print(f"Destination: {dest_folder}")

        # Get hash method from config, default to 'phash' if not specified
        hash_method = content_config.get('hashing', {}).get('method', 'phash')

        unique_count, duplicate_count = remove_duplicate_images(
            input_folder=source_folder,
            output_folder=dest_folder,
            hash_method=hash_method
        )

        print(f"Processed {unique_count + duplicate_count} total images")
        print(f"Copied {unique_count} unique images to {dest_folder}")
        print(f"Skipped {duplicate_count} duplicate images")

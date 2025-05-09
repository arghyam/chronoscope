import os
import re
import shutil

import yaml

def load_config():
    with open('src/data_cleaning/data_cleaning_config.yaml', 'r') as file:
        return yaml.safe_load(file)

def remove_duplicate_images(source_folder, destination_folder, allowed_extensions):
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Dictionary to store original filenames
    original_files = {}

    # Regular expression pattern to match files with (1), (2) etc.
    pattern = r'(.+?)(?:\(\d+\))?\.(jpg|jpeg|png)$'

    # Count variables
    original_count = 0
    final_count = 0

    # Iterate through all files in source folder
    for filename in os.listdir(source_folder):
        if any(filename.lower().endswith(ext) for ext in allowed_extensions):
            match = re.match(pattern, filename, re.IGNORECASE)
            if match:
                base_name = match.group(1)
                match.group(2)

                # If this is the original file (without numbers in parentheses)
                if '(' not in filename:
                    original_files[base_name] = filename
                    shutil.copy2(
                        os.path.join(source_folder, filename),
                        os.path.join(destination_folder, filename)
                    )

    original_count = len(os.listdir(source_folder))
    final_count = len(os.listdir(destination_folder))

    print(f"Original number of images: {original_count}")
    print(f"Number of images after removing duplicates: {final_count}")
    print(f"Number of duplicates removed: {original_count - final_count}")

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    duplicate_config = config['duplicate_removal']

    # Process each source-destination pair
    for source_folder, destination_folder in zip(
        duplicate_config['source_folders'],
        duplicate_config['destination_folders']
    ):
        print(f"\nProcessing folder: {source_folder}")
        remove_duplicate_images(
            source_folder,
            destination_folder,
            duplicate_config['allowed_extensions']
        )

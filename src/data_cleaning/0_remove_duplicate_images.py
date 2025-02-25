import os
import shutil
import re

def remove_duplicate_images(source_folder, destination_folder):
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
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            match = re.match(pattern, filename, re.IGNORECASE)
            if match:
                base_name = match.group(1)
                extension = match.group(2)
                
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
    # Define source and destination folders
    source_folder = "data/original_images"  # Change this to your source folder path
    destination_folder = "data/refined_data"  # This will create refined_data inside data folder
    
    remove_duplicate_images(source_folder, destination_folder)

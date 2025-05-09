import os
import shutil

import yaml
from sklearn.model_selection import train_test_split

def create_directory_structure(base_dir):
    """Create the directory structure for train, validation, and test sets."""
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)

    # Create train, val, test directories
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(base_dir, split, subdir), exist_ok=True)

    print(f"Created directory structure at {base_dir}")

def copy_files(file_list, src_img_dir, src_label_dir, dst_base_dir, split):
    """Copy image and label files to the destination directory."""
    for filename in file_list:
        # Get base filename without extension
        base_name = os.path.splitext(filename)[0]

        # Source paths
        img_src = os.path.join(src_img_dir, filename)
        label_src = os.path.join(src_label_dir, f"{base_name}.txt")

        # Destination paths
        img_dst = os.path.join(dst_base_dir, split, 'images', filename)
        label_dst = os.path.join(dst_base_dir, split, 'labels', f"{base_name}.txt")

        # Copy image if it exists
        if os.path.exists(img_src):
            shutil.copy2(img_src, img_dst)
        else:
            print(f"Warning: Image not found: {img_src}")

        # Copy label if it exists
        if os.path.exists(label_src):
            shutil.copy2(label_src, label_dst)
        else:
            print(f"Warning: Label not found: {label_src}")

def create_yaml_file(output_dir, class_names):
    """Create YAML configuration file for YOLO training."""
    yaml_content = {
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"Created YAML configuration file at {yaml_path}")

def split_dataset(src_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train, validation, and test sets."""
    # Source directories

    src_img_dir = os.path.join(src_dir, 'images')
    src_label_dir = os.path.join(src_dir, 'labels')

    # Get all image files
    image_files = [f for f in os.listdir(src_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Check if there are any images
    if not image_files:
        print(f"Error: No images found in {src_img_dir}")
        return

    # Verify that we have corresponding label files
    valid_files = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        if os.path.exists(os.path.join(src_label_dir, label_file)):
            valid_files.append(img_file)
        else:
            print(f"Warning: No label file for {img_file}")

    print(f"Found {len(valid_files)} valid image-label pairs")

    # Split the dataset
    train_files, temp_files = train_test_split(valid_files, test_size=(val_ratio + test_ratio), random_state=42)

    # Calculate the ratio for val and test from the temp set
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(temp_files, test_size=(1 - val_test_ratio), random_state=42)

    print(f"Split dataset: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")

    # Create directory structure
    create_directory_structure(output_dir)

    # Copy files to respective directories
    copy_files(train_files, src_img_dir, src_label_dir, output_dir, 'train')
    copy_files(val_files, src_img_dir, src_label_dir, output_dir, 'val')
    copy_files(test_files, src_img_dir, src_label_dir, output_dir, 'test')

    # Load class names
    classes_file = os.path.join(src_dir, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        print(f"Warning: Classes file not found: {classes_file}")
        class_names = ["meter"]  # Default class name

    # Create YAML file
    create_yaml_file(output_dir, class_names)

    # Copy classes.txt to output directory
    if os.path.exists(classes_file):
        shutil.copy2(classes_file, os.path.join(output_dir, 'classes.txt'))
    else:
        # Create classes.txt if it doesn't exist
        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            f.write('\n'.join(class_names))

    print(f"Dataset splitting complete. Files saved to {output_dir}")

def main():
    # Load configuration from YAML file
    with open('src/model_training/model_training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Get splitting configuration
    split_config = config['data_splitting']

    # Get paths from config
    src_dir = split_config['source']['base_dir']
    output_dir = split_config['output']['base_dir']

    # Get split ratios from config
    train_ratio = split_config['split_ratios']['train']
    val_ratio = split_config['split_ratios']['val']
    test_ratio = split_config['split_ratios']['test']

    # Split dataset
    split_dataset(src_dir, output_dir, train_ratio, val_ratio, test_ratio)

if __name__ == "__main__":
    main()

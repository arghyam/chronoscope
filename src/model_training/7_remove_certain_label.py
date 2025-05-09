import glob
import os

def remove_label_from_files(base_dir, label_to_remove='10'):
    """
    Removes rows containing a specific label from all label files in the directory structure.

    Args:
        base_dir: Path to directory containing train, valid, test folders
        label_to_remove: The label to remove from the files
    """
    # Process train, valid, and test directories
    for dataset_type in ['train', 'val', 'test']:
        labels_dir = os.path.join(base_dir, dataset_type, 'labels')

        # Skip if directory doesn't exist
        if not os.path.exists(labels_dir):
            print(f"Directory not found: {labels_dir}")
            continue

        # Get all label files
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        print(f"Found {len(label_files)} label files in {labels_dir}")

        # Process each file
        for label_file in label_files:
            # Read file content
            with open(label_file, 'r') as f:
                lines = f.readlines()

            # Filter out lines with label_to_remove
            filtered_lines = []
            for line in lines:
                # In YOLOv11 OBB format, the first value is typically the class id
                parts = line.strip().split()
                if parts and parts[0] != label_to_remove:
                    filtered_lines.append(line)

            # Write back filtered content
            with open(label_file, 'w') as f:
                f.writelines(filtered_lines)

        print(f"Processed labels in {dataset_type}")

# Example usage
if __name__ == "__main__":
    # Replace with your actual path
    base_directory = "data/cleaned_data/yolo_final_data_indivisual_numbers"

    # Remove label '10' from all label files
    remove_label_from_files(base_directory, label_to_remove='10')
    print("Completed removing label '10' from all files")

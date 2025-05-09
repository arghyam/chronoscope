import csv
import glob
import os

def load_yolo_obb_annotation(label_file):
    """
    Load YOLO OBB annotation and extract digit classes and coordinates.

    Args:
        label_file: Path to the YOLO OBB format label file

    Returns:
        Digit sequence sorted by x-coordinate (left to right)
    """
    annotations = []

    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if not parts or len(parts) < 9:  # Class ID + 8 coordinates
                continue

            class_id = int(parts[0])  # Digit class (0-9)

            # Extract all coordinates (x1,y1,x2,y2,x3,y3,x4,y4) - normalized values
            coords = [float(p) for p in parts[1:9]]

            # Extract all x-coordinates for the box
            x_coords = [coords[0], coords[2], coords[4], coords[6]]

            # Calculate center x-coordinate for sorting
            x_center = sum(x_coords) / 4

            annotations.append({
                'class_id': class_id,
                'x_center': x_center
            })

    # Sort annotations by x-coordinate (left to right)
    sorted_annotations = sorted(annotations, key=lambda x: x['x_center'])

    # Extract the digit sequence
    digit_sequence = ''.join(str(ann['class_id']) for ann in sorted_annotations)

    return digit_sequence

def process_labels_and_create_csv(images_dir, labels_dir, output_csv):
    """
    Process all label files and create a CSV with image names and digit sequences.

    Args:
        images_dir: Directory containing the images
        labels_dir: Directory containing the YOLO OBB format label files
        output_csv: Path to save the output CSV file
    """
    results = []

    # Get all label files
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))

    for label_file in label_files:
        # Get corresponding image filename
        label_filename = os.path.basename(label_file)
        base_name = os.path.splitext(label_filename)[0]

        # Try different image extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            image_filename = base_name + ext
            image_path = os.path.join(images_dir, image_filename)
            if os.path.exists(image_path):
                break
        else:
            # No matching image found with any extension
            image_filename = base_name + '.png'  # Default to png for consistency
        image_path = os.path.join(images_dir, image_filename)

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for label {label_filename}")
            continue

        # Process the label file to get digit sequence
        digit_sequence = load_yolo_obb_annotation(label_file)

        # Add to results
        results.append({
            'image_name': image_filename,
            'digit_sequence': digit_sequence
        })

    # Print some statistics
    print(f"Processed {len(results)} images with labels")

    if results:
        # Show some examples
        print("Sample results:")
        for i, result in enumerate(results[:5]):  # Show first 5 results
            print(f"  {result['image_name']}: {result['digit_sequence']}")

    # Write results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'digit_sequence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {output_csv}")

def main():
    # Paths
    images_dir = "dataset/training_data/yolo_indivisual/test/images"
    labels_dir = "dataset/training_data/yolo_indivisual/test/labels"
    output_csv = "dataset/inference_data_check/test_images/digit_sequences.csv"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Process labels and create CSV
    process_labels_and_create_csv(images_dir, labels_dir, output_csv)

if __name__ == "__main__":
    main()

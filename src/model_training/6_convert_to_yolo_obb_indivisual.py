import os
import shutil
import xml.etree.ElementTree as ET

import yaml

def parse_xml_annotations(xml_file):
    """Parse XML file and extract annotations."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []

    # Iterate through all image elements
    for image in root.findall('.//image'):
        img_id = image.get('id')
        img_name = image.get('name')
        img_width = int(image.get('width'))
        img_height = int(image.get('height'))

        # Check for polygon or box annotations
        polygons = image.findall('.//polygon')
        boxes = image.findall('.//box')

        img_annotations = []

        # Process polygons
        for polygon in polygons:
            label = polygon.get('label')
            # Skip unclear image labels
            if label == 'unclear image':
                continue

            points_str = polygon.get('points')
            points = [float(p) for p in points_str.replace(';', ',').split(',')]

            # Ensure we have 4 points (8 coordinates)
            if len(points) == 8:
                img_annotations.append({
                    'label': label,
                    'points': points,
                    'type': 'polygon'
                })

        # Process boxes
        for box in boxes:
            label = box.get('label')
            # Skip unclear image labels
            if label == 'unclear image':
                continue

            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            # Convert box to 4 points (clockwise: top-left, top-right, bottom-right, bottom-left)
            points = [
                xtl, ytl,  # top-left
                xbr, ytl,  # top-right
                xbr, ybr,  # bottom-right
                xtl, ybr   # bottom-left
            ]

            img_annotations.append({
                'label': label,
                'points': points,
                'type': 'box'
            })

        # Only add images that have valid annotations (not unclear)
        if img_annotations:
            annotations.append({
                'id': img_id,
                'name': img_name,
                'width': img_width,
                'height': img_height,
                'annotations': img_annotations
            })

    return annotations

def convert_to_yolo_obb(annotation, label_map):
    """Convert annotation to YOLO OBB format."""
    img_width = annotation['width']
    img_height = annotation['height']

    yolo_annotations = []

    for ann in annotation['annotations']:
        label = ann['label']
        if label not in label_map:
            # Skip annotations with labels not in our predefined map
            continue

        class_id = label_map[label]
        points = ann['points']

        # Normalize coordinates
        normalized_points = []
        for i in range(0, len(points), 2):
            x = points[i] / img_width
            y = points[i+1] / img_height
            normalized_points.extend([x, y])

        # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
        yolo_line = f"{class_id} " + " ".join([f"{p:.6f}" for p in normalized_points])
        yolo_annotations.append(yolo_line)

    return yolo_annotations

def create_yolo_dataset(annotations, source_dir, output_dir, label_map):
    """Create YOLO dataset with images and labels."""
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    processed_count = 0

    for annotation in annotations:
        img_name = annotation['name']
        img_path = os.path.join(source_dir, img_name)

        # Skip if source image doesn't exist
        if not os.path.exists(img_path):
            print(f"Warning: Source image not found: {img_path}")
            continue

        # Copy image to output directory
        shutil.copy(img_path, os.path.join(images_dir, img_name))

        # Create label file
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)

        # Convert annotations to YOLO OBB format
        yolo_annotations = convert_to_yolo_obb(annotation, label_map)

        # Write label file only if there are valid annotations
        if yolo_annotations:
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            processed_count += 1

    # Save label map
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for label, class_id in sorted(label_map.items(), key=lambda x: x[1]):
            f.write(f"{label}\n")

    print(f"Processed {processed_count} images")
    print(f"Label map: {label_map}")
    print(f"Dataset created at: {output_dir}")

    return label_map

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'model_training_config.yaml')
    config = load_config(config_path)

    # Extract paths from config
    xml_file = config['data_conversion']['xml_file']
    source_dir = config['data_conversion']['source_dir']
    output_dir = config['data_conversion']['output_dir']
    label_map = config['label_mapping']

    # Parse XML annotations
    print("Parsing XML annotations...")
    annotations = parse_xml_annotations(xml_file)
    print(f"Found {len(annotations)} annotated images (excluding unclear images)")

    # Create YOLO dataset
    print("Creating YOLO dataset...")
    label_map = create_yolo_dataset(annotations, source_dir, output_dir, label_map)

    # Display stats about the dataset
    print("\nLabel Statistics:")
    for label, class_id in sorted(label_map.items(), key=lambda x: x[1]):
        count = sum(1 for ann in annotations for obj in ann['annotations'] if obj['label'] == label)
        print(f"  Class {class_id} ({label}): {count} instances")

    print("\nDone!")

if __name__ == "__main__":
    main()

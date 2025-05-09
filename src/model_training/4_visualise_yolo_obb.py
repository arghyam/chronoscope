import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import yaml

def load_config(config_path='src/model_training/model_training_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_class_names(classes_file):
    """Load class names from file."""
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def load_yolo_obb_annotation(label_file, img_width, img_height):
    """Load YOLO OBB annotation and convert to pixel coordinates."""
    annotations = []

    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])

            # Extract normalized coordinates
            coords = [float(p) for p in parts[1:]]

            # Convert normalized coordinates to pixel coordinates
            pixel_coords = []
            for i in range(0, len(coords), 2):
                x = coords[i] * img_width
                y = coords[i+1] * img_height
                pixel_coords.append((int(x), int(y)))

            annotations.append({
                'class_id': class_id,
                'points': pixel_coords
            })

    return annotations

def get_color_for_class(class_id, class_name, config):
    """Get color for visualization based on class."""
    colors = config['visualization']['colors']
    if class_name.lower().startswith('meter'):
        return colors['meter']
    return colors['digits']  # Default color for digit classes

def visualize_yolo_obb(image_path, label_path, output_path, class_names, config):
    """Visualize YOLO OBB annotations on image and save to output path."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    img_height, img_width = img.shape[:2]

    # Load annotations
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return

    annotations = load_yolo_obb_annotation(label_path, img_width, img_height)

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create figure and axis
    fig_size = config['visualization']['settings']['figure_size']
    fig, ax = plt.subplots(1, figsize=tuple(fig_size))
    ax.imshow(img_rgb)

    # Draw bounding boxes
    for ann in annotations:
        class_id = ann['class_id']
        points = ann['points']

        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        color = get_color_for_class(class_id, class_name, config)

        # Create polygon patch
        polygon = patches.Polygon(
            points,
            linewidth=config['visualization']['settings']['bbox_linewidth'],
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(polygon)

        # Add class label
        ax.text(
            points[0][0], points[0][1] - 10,
            class_name,
            bbox=dict(facecolor=color, alpha=0.5),
            fontsize=config['visualization']['settings']['font_size'],
            color='white'
        )

    # Remove axis
    plt.axis('off')

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Saved visualization to {output_path}")

def main():
    # Load configuration
    config = load_config()

    # Get paths from config
    yolo_data_dir = config['visualization']['data_paths']['yolo_data_dir']
    output_dir = config['visualization']['data_paths']['output_dir']

    # Set up directories
    images_dir = os.path.join(yolo_data_dir, 'images')
    labels_dir = os.path.join(yolo_data_dir, 'labels')
    classes_file = os.path.join(yolo_data_dir, 'classes.txt')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load class names
    if os.path.exists(classes_file):
        class_names = load_class_names(classes_file)
    else:
        print(f"Warning: Classes file not found: {classes_file}")
        class_names = ["Unknown"]

    # Process all images
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(image_files)} images. Starting visualization...")

    for img_file in image_files:
        image_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_visualized.jpg")

        visualize_yolo_obb(image_path, label_path, output_path, class_names, config)

    print(f"Visualization complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()

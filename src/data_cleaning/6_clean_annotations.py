import os
import pandas as pd
from pathlib import Path

def clean_annotations(annotations_path, images_folder):
    """
    Clean annotations CSV by removing entries for images that don't exist in the images folder.
    
    Args:
        annotations_path: Path to the annotations CSV file
        images_folder: Path to the folder containing images
    
    Returns:
        DataFrame with only annotations for existing images
    """
    # Read the annotations file
    print(f"Reading annotations from {annotations_path}")
    annotations_df = pd.read_csv(annotations_path)
    
    # Get the initial count of annotations
    initial_count = len(annotations_df)
    print(f"Initial annotations count: {initial_count}")
    
    # Get the image filename column (assuming it's the first column, adjust if needed)
    # You may need to change this based on your CSV structure
    image_column = annotations_df.columns[0]
    
    # Check which images exist
    existing_images = []
    missing_images = []
    
    for image_name in annotations_df[image_column]:
        image_path = os.path.join(images_folder, image_name)
        if os.path.exists(image_path):
            existing_images.append(image_name)
        else:
            missing_images.append(image_name)
    
    # Filter annotations to keep only those with existing images
    clean_df = annotations_df[annotations_df[image_column].isin(existing_images)]
    
    # Print statistics
    print(f"Found {len(existing_images)} existing images")
    print(f"Found {len(missing_images)} missing images")
    print(f"Removed {initial_count - len(clean_df)} annotations")
    
    if missing_images:
        print("First few missing images:")
        for img in missing_images[:5]:
            print(f"  - {img}")
        if len(missing_images) > 5:
            print(f"  ... and {len(missing_images) - 5} more")
    
    return clean_df

if __name__ == "__main__":
    # Paths
    annotations_path = "data/data_suman/annotations.csv"
    images_folder = "data/refined_data_3"
    
    # Clean the annotations
    clean_df = clean_annotations(annotations_path, images_folder)
    
    # Save the cleaned annotations
    output_path = "annotations_cleaned.csv"
    clean_df.to_csv(output_path, index=False)
    print(f"Cleaned annotations saved to {output_path}")

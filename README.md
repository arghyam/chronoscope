# Arghyam-BFM-Reading
Bulk Flow Meter reading extraction for Arghyam

# Stage 1: Data Cleaning

The first phase of the project is about data cleaning and removing duplicate images. This stage consists of two main steps:

## 1. Duplicate Image Removal
Script: `src/data_cleaning/0_remove_duplicate_images.py`

This script handles the removal of duplicate images from the dataset. It:
- Identifies and removes duplicate images that have the same base name with numbering (e.g., image.jpg, image(1).jpg)
- Keeps only the original version of each image
- Copies cleaned images to a new directory

Usage:
```bash
python src/data_cleaning/0_remove_duplicate_images.py
```

Configuration:
- Source folder: `data/original_images` - Directory containing the original dataset
- Destination folder: `data/refined_data` - Directory where cleaned images will be stored

The script will output:
- Original number of images
- Number of images after removing duplicates
- Number of duplicates removed

## 2. Image Quality Assessment
Script: `src/data_cleaning/1_streamlit_data_quality.py`

This interactive Streamlit application allows for manual quality assessment of the cleaned images. It helps identify and annotate images based on their quality characteristics.

Features:
- Interactive web interface for image annotation
- Six quality categories: Good, Blurry, Out of focus, Oriented, Foggy, Poor lighting
- Automatic progress saving
- Resume capability for interrupted annotation sessions

Usage:
```bash
streamlit run src/data_cleaning/1_streamlit_data_quality.py
```

The tool:
- Displays images one at a time
- Allows quick annotation through button clicks
- Saves annotations automatically to a CSV file (`annotations.csv`) in the image folder
- Tracks progress and allows resuming from where you left off

Output:
- Creates an `annotations.csv` file with columns:
  - image_name: Name of the image file
  - annotation: Selected quality category
  - annotation_done: Annotation status

This two-step process ensures that the dataset is free from duplicates and contains only images of acceptable quality for further processing.



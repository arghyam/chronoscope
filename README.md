# Arghyam-BFM-Reading
Bulk Flow Meter reading extraction for Arghyam

# Project Structure

This project is organized into two main stages:
1. Data Cleaning
2. Model Training

# Stage 1: Data Cleaning

The first phase of the project is about data cleaning and preparing the dataset. This stage consists of several steps:

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

## 3. Data Distribution Analysis
Script: `src/data_cleaning/2_data_distribution.py`

This script analyzes the distribution of image annotations from the quality assessment process. It provides a statistical overview of the dataset quality.

Features:
- Reads the annotations CSV file generated by the quality assessment tool
- Calculates the distribution of images across different quality categories
- Displays the count and percentage for each annotation category

Usage:
```bash
python src/data_cleaning/2_data_distribution.py
```

Output:
- Prints a summary of annotation distribution to the console
- Shows both count and percentage for each quality category

## 4. Perceptual Duplicate Detection
Script: `src/data_cleaning/4_removing_duplicate_data_same_images.py`

This script identifies and removes visually duplicate images using perceptual hashing techniques. Unlike the first duplicate removal script, this one detects images that look similar even if they have different filenames.

Features:
- Multiple hash methods available: perceptual hash (phash), difference hash (dhash), average hash (ahash), and MD5
- Identifies visually similar images regardless of filename
- Preserves only one copy of each unique image

Usage:
```bash
python src/data_cleaning/4_removing_duplicate_data_same_images.py
```

Configuration:
- Input folder: `data/original_images_2` - Directory containing images to check for duplicates
- Output folder: `data/refined_data_2` - Directory where unique images will be stored
- Hash method: 'phash' (default) - Perceptual hash algorithm for image comparison

The script will output:
- Total number of images processed
- Number of unique images identified and copied
- Number of duplicate images skipped

## 5. Image Orientation Detection
Script: `src/data_cleaning/5_checking_orientation_images.py`

This script analyzes images to determine if they are properly oriented or rotated. It uses multiple methods including EXIF metadata and OCR-based detection to identify images that may need rotation.

Features:
- EXIF metadata analysis to check for orientation flags
- OCR-based orientation detection using text recognition
- Multiple detection methods with fallback options
- Comprehensive CSV report generation

Usage:
```bash
python src/data_cleaning/5_checking_orientation_images.py --input_dir [input_directory] --output_csv [output_file.csv]
```

Parameters:
- `--input_dir`: Directory containing images to check for orientation
- `--output_csv`: Path to save the CSV report

The script will output:
- A CSV file with columns:
  - image_name: Name of the image file
  - is_properly_oriented: 'Yes' or 'No' indicating orientation status
- Progress updates during processing

## 6. Image Orientation Correction
Script: `src/data_cleaning/5_correcting_orientations.py`

This script corrects the orientation of images identified as improperly oriented by the previous script.

Features:
- Automatic detection of image orientation using multiple methods
- Correction of rotated images to proper orientation
- Support for various rotation angles (90°, 180°, 270°)
- Preservation of image quality during rotation

Usage:
```bash
python src/data_cleaning/5_correcting_orientations.py --input_dir [input_directory] --output_dir [output_directory]
```

Parameters:
- `--input_dir`: Directory containing images to correct
- `--output_dir`: Directory where corrected images will be saved

## 7. Annotation Cleaning
Script: `src/data_cleaning/6_clean_annotations.py`

This script cleans the annotations CSV file by removing entries for images that no longer exist in the dataset (e.g., after duplicate removal).

Features:
- Verification of image existence for each annotation
- Removal of annotations for missing images
- Generation of a cleaned annotations file

Usage:
```bash
python src/data_cleaning/6_clean_annotations.py
```

The script will output:
- Initial count of annotations
- Number of missing images detected
- Final count of valid annotations
- Saves a new cleaned annotations file (`annotations_cleaned.csv`)

## 8. Good Image Selection
Script: `src/data_cleaning/7_copy_good_images.py`

This script selects and copies only the images annotated as "Good" quality to a final dataset directory.

Features:
- Filters images based on quality annotations
- Copies only high-quality images to the final dataset
- Maintains original image quality during copying

Usage:
```bash
python src/data_cleaning/7_copy_good_images.py
```

The script will:
- Create a `data/final_data` directory
- Copy all images annotated as "Good" to this directory
- Report the total number of good images copied

# Stage 2: Model Training

The second phase of the project focuses on training a model to detect and read bulk flow meters.

## 1. Meter Perspective Transform
Script: `src/model_training/1_meter_perspective_transform.py`

This script applies perspective transformation to straighten oriented bounding box regions in images, which is useful for normalizing meter displays before OCR.

Features:
- Straightens oriented bounding boxes using perspective transformation
- Normalizes meter displays for better OCR results
- Preserves aspect ratio of the meter region

Usage:
```bash
python src/model_training/1_meter_perspective_transform.py
```

## 2. OCR with Mistral AI
Script: `src/model_training/2_mistral_ocr.py`

This script uses Gemini 2.0 Flash (via Google's Generative AI) to perform OCR on meter images and extract readings.

Features:
- Base64 encoding of images for API submission
- Integration with Google's Generative AI
- Extraction of meter readings from images
- Error handling for API requests

Usage:
```bash
python src/model_training/2_mistral_ocr.py
```

Requirements:
- Google API key (set in .env file)
- Internet connection for API access

## 3. YOLO OBB Format Conversion
Script: `src/model_training/3_convert_to_yolo_obb_format.py`

This script converts annotations from XML format to YOLO OBB (Oriented Bounding Box) format for training object detection models.

Features:
- Parses XML annotations (CVAT format)
- Converts to YOLO OBB format (class_id, x1, y1, x2, y2, x3, y3, x4, y4)
- Creates a properly structured dataset for YOLO training
- Handles both polygon and box annotations

Usage:
```bash
python src/model_training/3_convert_to_yolo_obb_format.py
```

## 4. YOLO OBB Visualization
Script: `src/model_training/4_visualise_yolo_obb.py`

This script visualizes YOLO OBB annotations on images to verify correct conversion and annotation quality.

Features:
- Loads YOLO OBB annotations and converts to pixel coordinates
- Visualizes oriented bounding boxes on images
- Displays class labels and confidence scores
- Saves annotated images for review

Usage:
```bash
python src/model_training/4_visualise_yolo_obb.py
```

## 5. Dataset Splitting for YOLO
Script: `src/model_training/5_data_splitting_yolo.py`

This script splits the dataset into training, validation, and test sets for YOLO model training.

Features:
- Creates appropriate directory structure for YOLO training
- Splits dataset with configurable ratios (default: 80% train, 10% validation, 10% test)
- Ensures paired image and label files are kept together
- Generates YAML configuration file for YOLO training

Usage:
```bash
python src/model_training/5_data_splitting_yolo.py
```

Output:
- Creates train, validation, and test directories with images and labels
- Generates a YAML configuration file with dataset paths and class names

This comprehensive pipeline ensures that the dataset is properly cleaned, annotated, and prepared for training a model to detect and read bulk flow meters.

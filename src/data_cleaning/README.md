# Chronoscope Data Cleaning Documentation

## Script 1: Remove Duplicate Images (`0_remove_duplicate_images.py`)

### Purpose
Script to remove duplicate images that have been automatically renamed with patterns like `filename(1).jpg`, `filename(2).jpg`. Keeps only the original version (without numbers in parentheses).

### Configuration (`data_cleaning_config.yaml`)
```yaml
duplicate_removal:
  source_folders:
    - "dataset/original_data/dataset1"
  destination_folders:
    - "dataset/data_cleaned/removing_duplicates/dataset4"
  allowed_extensions:
    - ".jpg"
    - ".jpeg"
    - ".png"
```
### Original data is in this s3 bucket
- Console Link: [AWS S3 Chronodata Bucket](https://ap-south-1.console.aws.amazon.com/s3/buckets/chronodata?region=ap-south-1&bucketType=general&prefix=dataset/original_data/&showversions=false)
```
s3://chronodata/dataset/original_data/
```

### How It Works
1. Reads configuration from YAML file
2. Scans source directory for images
3. Uses regex to identify and filter duplicate files
4. Copies original files (without numbered suffixes) to destination
5. Reports statistics about duplicates removed

### Usage
```bash

# Run script
python src/data_cleaning/0_remove_duplicate_images.py
```

## Script 2: Content-Based Duplicate Image Removal (`1_removing_duplicate_data_same_images.py`)

### Purpose
Script to remove duplicate images based on their visual content using perceptual hashing. This detects and removes duplicate images even if they have different filenames or minor modifications.

### Configuration (`data_cleaning_config.yaml`)
```yaml
duplicate_removal:
  content_based:
    source_folders:
      - "dataset/original_data/dataset1"
    destination_folders:
      - "dataset/data_cleaned/removing_duplicates/dataset1"
    allowed_extensions:
      - ".jpg"
      - ".jpeg"
      - ".png"
    hashing:
      method: "phash"  # Options: 'phash', 'dhash', 'ahash', 'md5'
```

### Original data is in this s3 bucket
- Input data Console Link: [AWS S3 Chronodata Bucket](https://ap-south-1.console.aws.amazon.com/s3/buckets/chronodata?region=ap-south-1&bucketType=general&prefix=dataset/data_cleaned/removing_duplicates/&showversions=false)
- Output data Console Link: [AWS S3 Chronodata Bucket](https://ap-south-1.console.aws.amazon.com/s3/buckets/chronodata?region=ap-south-1&bucketType=general&prefix=dataset/data_cleaned/dedup/&showversions=false)```

## Script 3: Cross-Dataset Similar Image Removal (`3_removing_similar_images.py`)

### Purpose
Script to identify and copy unique images that exist in one dataset but not in another. This is useful for maintaining distinct image sets and avoiding duplicates across different datasets.

### Configuration (`data_cleaning_config.yaml`)
```yaml
duplicate_removal:
  cross_dataset_comparison:
    reference_dataset: "dataset/data_cleaned/dedup/dataset1"
    comparison_dataset: "dataset/data_cleaned/dedup/dataset4"
    destination_folder: "dataset/filtered_data/dedup/dataset2"
    allowed_extensions:
      - ".jpg"
      - ".jpeg"
      - ".png"
```

### How It Works
1. Loads configuration from YAML file
2. Creates the destination directory if it doesn't exist
3. Compares image lists between reference and comparison datasets
4. Identifies images that are unique to the comparison dataset
5. Copies unique images to the destination folder
6. Reports statistics about the number of unique images copied

### Usage
```bash
# Run script
python src/data_cleaning/3_removing_similar_images.py
```

### Data Flow
- Input: Two datasets (reference and comparison) from deduplication stage
- Output: A new dataset containing only unique images from the comparison dataset
- Location: The results are stored in the configured destination folder

### Original data locations
- Reference and Comparision  data Console Link: [AWS S3 Chronodata Bucket](https://ap-south-1.console.aws.amazon.com/s3/buckets/chronodata?region=ap-south-1&bucketType=general&prefix=dataset/data_cleaned/dedup/&showversions=false)
- Output data Console Link: [AWS S3 Chronodata Bucket](https://ap-south-1.console.aws.amazon.com/s3/buckets/chronodata?region=ap-south-1&bucketType=general&prefix=dataset/data_cleaned/filtered_data/&showversions=false)


## Script 4: Image Quality Annotation Tool (`1_streamlit_data_quality.py`)

### Purpose
A Streamlit-based interactive web application for manually annotating image quality issues. This tool allows users to systematically review images and categorize them based on various quality parameters such as blurriness, focus, orientation, lighting conditions, etc.

### Configuration
This script doesn't require a YAML configuration file. Instead, it uses direct user input through the Streamlit interface.

### Features
- Interactive web interface for image annotation
- Six quality categories:
  - Good
  - Blurry
  - Out of focus
  - Oriented
  - Foggy
  - Poor lighting
- Automatic progress saving
- Resume capability from last annotation session
- Real-time image preview
- Progress tracking

### How It Works
1. **Initialization**
   - Creates a Streamlit web interface
   - Initializes session state for tracking progress
   - Loads existing annotations if available

2. **Image Loading**
   - Accepts folder path input from user
   - Supports multiple image formats (.png, .jpg, .jpeg, .gif, .bmp)
   - Automatically finds first unannotated image

3. **Annotation Process**
   - Displays current image in full size
   - Shows progress (current image number / total images)
   - Provides clickable buttons for quality categories
   - Saves annotations automatically after each selection

4. **Data Storage**
   - Creates/updates 'annotations.csv' in the image folder
   - CSV structure:
     ```csv
     image_name,annotation,annotation_done
     ```

### Usage
```bash
# Start the Streamlit application
streamlit run src/data_cleaning/1_streamlit_data_quality.py
```

### Interface Instructions
1. Launch the application using the command above
2. Enter the full path to your images folder in the sidebar
3. For each displayed image:
   - Review the image quality
   - Select appropriate quality category
   - Click "Save and Next" to proceed
4. Annotations are automatically saved after each image

### Data Flow
- Input: Directory containing images to be annotated
- Output: annotations.csv file containing:
  - Image filenames
  - Quality annotations
  - Annotation completion status
- Location: CSV file is stored in the same directory as the images

### Notes
- The tool automatically resumes from the last unannotated image
- Annotations can be modified by revisiting the folder
- Progress is saved automatically to prevent data loss
- Images are displayed in their original orientation using OpenCV

## Script 5: Data Distribution Analysis (`2_data_distribution.py`)

### Purpose
Script to analyze and visualize the distribution of image quality annotations. This tool processes the annotations generated by the quality annotation tool and provides statistical insights about the distribution of different quality categories in the dataset.

### Configuration
This script uses a direct file path rather than a YAML configuration. The default path is:
```python
'data/data_suman/annotations.csv'
```

### How It Works
1. Reads the annotations CSV file generated by the quality annotation tool
2. Calculates the distribution of annotations across different quality categories
3. Computes percentage distribution for each category
4. Displays a detailed breakdown of:
   - Count per category
   - Percentage distribution per category

### Usage
```bash
# Run script
python src/data_cleaning/2_data_distribution.py
```

### Data Flow
- Input: annotations.csv file containing quality annotations
- Output: Console display showing:
  - Distribution counts for each quality category
  - Percentage distribution for each category
- Format of output:
  ```
  Distribution of annotations:
  Category: Count (Percentage%)
  ```

### Example Output
Distribution of annotations:
Good: 150 (30.0%)
Blurry: 75 (15.0%)
Out of focus: 100 (20.0%)
Oriented: 50 (10.0%)
Foggy: 75 (15.0%)
Poor lighting: 50 (10.0%)


### Notes
- This script is designed to work with the output from the Image Quality Annotation Tool
- Provides quick insights into the quality distribution of the dataset
- Useful for identifying quality trends and potential biases in the dataset

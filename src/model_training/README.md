# Chronoscope Model Training Documentation

## Script 1: YOLO OBB Format Converter (`6_convert_to_yolo_obb_indivisual.py`)

### Purpose
Script to convert XML annotations (from CVAT format) to YOLO OBB (Oriented Bounding Box) format. This conversion is necessary for training object detection models that can handle rotated bounding boxes, particularly useful for meter detection in various orientations.

All the annotated dat are in this s3 path
- Annotated data Console Link: [AWS S3 Chronodata Bucket](https://ap-south-1.console.aws.amazon.com/s3/buckets/chronodata?region=ap-south-1&bucketType=general&prefix=dataset/annotated_data/&showversions=false)


### Configuration (`model_training_config.yaml`)
```yaml
data_conversion:
  xml_file: 'dataset/annotated_data/dataset4_2688/dataset4/annotations.xml'
  source_dir: 'dataset/annotated_data/dataset4_2688/dataset4/images'
  output_dir: 'dataset/training_data/broader_meter/dataset4_2688'

label_mapping:
  meter: '0'
  # Uncomment and modify if needed for digit detection
  # '0': 0
  # '1': 1
  # '2': 2
  # '3': 3
  # '4': 4
  # '5': 5
  # '6': 6
  # '7': 7
  # '8': 8
  # '9': 9
```

### How It Works
1. **XML Parsing (`parse_xml_annotations`)**
   - Reads XML file containing CVAT annotations
   - Processes both polygon and box annotations
   - Filters out 'unclear image' labels
   - Converts box annotations to 4-point format
   - Returns structured annotation data

2. **YOLO OBB Conversion (`convert_to_yolo_obb`)**
   - Converts annotations to YOLO OBB format
   - Normalizes coordinates (0-1 range)
   - Format: `class_id x1 y1 x2 y2 x3 y3 x4 y4`
   - Handles both polygon and box annotations

3. **Dataset Creation (`create_yolo_dataset`)**
   - Creates output directory structure
   - Copies images to output directory
   - Generates label files in YOLO OBB format
   - Creates classes.txt with label mapping
   - Reports processing statistics

### Usage
```bash

# Run script
python src/model_training/6_convert_to_yolo_obb_indivisual.py
```

### Data Flow
- **Input**:
  - XML file containing CVAT annotations
  - Source directory containing original images
  - Configuration file with paths and label mapping
- **Output**:
  - Converted dataset in YOLO OBB format
  - Directory structure:
    ```
    output_dir/
    ├── images/
    │   └── [copied image files]
    ├── labels/
    │   └── [YOLO format .txt files]
    └── classes.txt
    ```

### File Formats
1. **Input XML Format (CVAT)**
   ```xml
   <annotations>
     <image id="0" name="image1.jpg" width="1920" height="1080">
       <polygon label="meter" points="x1,y1;x2,y2;x3,y3;x4,y4"/>
       <box label="meter" xtl="100" ytl="100" xbr="200" ybr="200"/>
     </image>
   </annotations>
   ```

2. **Output YOLO OBB Format**
   ```text
   # Each line in label file:
   class_id x1 y1 x2 y2 x3 y3 x4 y4
   ```
   Where:
   - class_id: Integer class identifier
   - x1,y1 to x4,y4: Normalized coordinates (0-1) of the 4 corners

### Notes
- Supports both polygon and rectangular box annotations
- Automatically skips unclear or invalid annotations
- Normalizes coordinates to handle different image sizes
- Creates a standardized dataset structure for YOLO training
- Generates statistics about processed images and class distribution

## Script 2: YOLO OBB Visualization (`4_visualise_yolo_obb.py`)

### Purpose
Script to visualize YOLO OBB (Oriented Bounding Box) annotations on images. This visualization tool helps in verifying the correctness of annotations and understanding the detection results by drawing rotated bounding boxes and class labels on the original images.

### Configuration (`model_training_config.yaml`)
```yaml
visualization:
  # Data paths for visualization
  data_paths:
    yolo_data_dir: 'dataset/training_data/indivisual_numbers/dataset1_1460'
    output_dir: 'dataset/data_quality_check/indivisual_numbers/dataset1_1460'

  # Visualization settings
  settings:
    figure_size: [12, 12]
    bbox_linewidth: 2
    font_size: 12

  # Color scheme for different classes
  colors:
    meter: 'red'
    digits: 'blue'  # Default color for all digit classes
```

### How It Works
1. **Configuration Loading (`load_config`)**
   - Reads YAML configuration file
   - Sets up visualization parameters and paths
   - Configures colors and display settings

2. **Annotation Processing (`load_yolo_obb_annotation`)**
   - Reads YOLO OBB format annotations
   - Converts normalized coordinates (0-1) to pixel coordinates
   - Structures annotations with class IDs and point coordinates

3. **Visualization (`visualize_yolo_obb`)**
   - Draws rotated bounding boxes using matplotlib
   - Adds class labels with colored backgrounds
   - Handles different classes with distinct colors
   - Saves annotated images to output directory

### Usage
```bash
# Run visualization script
python src/model_training/4_visualise_yolo_obb.py
```

### Data Flow
- **Input**:
  - YOLO format dataset directory containing:
    - Images directory with source images
    - Labels directory with YOLO OBB annotations
    - classes.txt with class mapping
  - Configuration file with visualization settings
- **Output**:
  - Visualized images with:
    - Rotated bounding boxes
    - Class labels
    - Color-coded annotations
  - Directory structure:
    ```
    output_dir/
    └── [original_filename]_visualized.jpg
    ```

### File Formats
1. **Input YOLO OBB Format**
   ```text
   # Each line in label file:
   class_id x1 y1 x2 y2 x3 y3 x4 y4
   ```
   Where:
   - class_id: Integer class identifier
   - x1,y1 to x4,y4: Normalized coordinates (0-1) of the 4 corners

2. **Classes File Format**
   ```text
   # Each line in classes.txt:
   class_name
   ```

### Notes
- Supports visualization of both meter and digit annotations
- Configurable colors and visualization settings
- Automatically processes all images in the dataset
- Preserves aspect ratio of original images
- Provides visual feedback for annotation verification
- Handles missing files and invalid annotations gracefully

## Script 3: YOLO Dataset Splitter (`5_data_splitting_yolo.py`)

### Purpose
Script to split a YOLO format dataset into train, validation, and test sets. This splitting tool helps in creating properly structured datasets for training object detection models, ensuring a good distribution of data across different sets while maintaining the paired relationship between images and their corresponding label files.

### Configuration (`model_training_config.yaml`)
```yaml
data_splitting:
  source:
    base_dir: 'dataset/training_data/individual_numbers_2f'
  output:
    base_dir: 'dataset/training_data/yolo_individual_numbers_2f'
  split_ratios:
    train: 0.8
    val: 0.1
    test: 0.1
```

### How It Works
1. **Directory Structure Creation (`create_directory_structure`)**
   - Creates base output directory
   - Sets up train, validation, and test subdirectories
   - Creates separate image and label folders for each split
   - Ensures all necessary directories exist

2. **Dataset Splitting (`split_dataset`)**
   - Identifies valid image-label pairs
   - Performs stratified splitting using sklearn
   - Maintains dataset distribution across splits
   - Handles missing files gracefully
   - Creates YOLO-compatible directory structure

3. **File Management (`copy_files`)**
   - Copies images and corresponding labels to appropriate directories
   - Maintains file relationships
   - Verifies file existence before copying
   - Reports any missing files

4. **YAML Configuration (`create_yaml_file`)**
   - Generates YOLO-compatible dataset configuration
   - Specifies paths for train, validation, and test sets
   - Includes class information and names
   - Creates standardized format for training

### Usage
```bash
# Run splitting script
python src/model_training/5_data_splitting_yolo.py
```

### Data Flow
- **Input**:
  - Source directory containing:
    - Images directory with source images
    - Labels directory with YOLO format annotations
    - classes.txt with class mapping
  - Configuration file with splitting parameters
- **Output**:
  - Split dataset with structure:
    ```
    output_dir/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    ├── classes.txt
    └── data.yaml
    ```

### File Formats
1. **Input Directory Structure**
   ```text
   source_dir/
   ├── images/
   │   └── [image files (.jpg, .jpeg, .png)]
   ├── labels/
   │   └── [YOLO format .txt files]
   └── classes.txt
   ```

2. **Output YAML Configuration**
   ```yaml
   train: train/images
   val: val/images
   test: test/images
   nc: [number of classes]
   names: [class names mapping]
   ```

### Notes
- Maintains consistent random state for reproducible splits
- Verifies image-label pair existence before processing
- Automatically handles missing files with warnings
- Creates YOLO-compatible dataset structure
- Supports configurable split ratios
- Preserves class distribution across splits
- Copies class information to output directory
- Generates ready-to-use YAML configuration for training


### Final data fro training for yolo object detection

Yolo training data for individual number detection :
[AWS S3 Chronodata Bucket](https://ap-south-1.console.aws.amazon.com/s3/buckets/chronodata?region=ap-south-1&bucketType=general&prefix=dataset/training_data/yolo_indivisual/&showversions=false)


Yolo training data for broad number detection :
[AWS S3 Chronodata Bucket](https://ap-south-1.console.aws.amazon.com/s3/buckets/chronodata?region=ap-south-1&bucketType=general&prefix=dataset/training_data/yolo_broader_meter/&showversions=false)

## Script 4: YOLO OBB Training (`train.py`)

### Purpose
Script to train a YOLO (You Only Look Once) model with OBB (Oriented Bounding Box) support using the Ultralytics framework. This training script is designed to train models that can detect objects with rotated bounding boxes, particularly useful for meter and digit detection at various orientations.

### Configuration
```yaml
training:
  # Model configuration
  model:
    base_model: 'yolov8m-obb.pt'
    task: 'obb'
    image_size: 640

  # Training parameters
  parameters:
    epochs: 100
    batch_size: 32

  # Server training specific parameters (optional)
  server_augmentations:
    augment: false
    fliplr: 0.0
    flipud: 0.0
    mosaic: 0.0
    mixup: 0.0
    scale: 0.0
```

### How It Works
1. **Model Initialization (`YOLO`)**
   - Loads pre-trained YOLO model (yolov8m-obb.pt)
   - Configures model for OBB detection task
   - Sets up training parameters

2. **Training Process (`model.train`)**
   - Executes training loop with specified parameters
   - Handles data loading and batching
   - Performs model optimization
   - Tracks training metrics

3. **Optional Operations**
   - Saves training results for analysis
   - Performs model validation
   - Generates performance metrics

### Usage
```bash
# Run training script
python src/model_training/train.py
```

### Data Flow
- **Input**:
  - Pre-trained YOLO model (yolov8m-obb.pt)
  - Dataset configuration (data.yaml)
  - Training parameters
- **Output**:
  - Trained model weights
  - Training metrics and results
  - Optional validation metrics

### Training Configurations

1. **Basic Training Setup**
   ```python
   from ultralytics import YOLO

   model = YOLO('yolov8m-obb.pt')
   results = model.train(
       task='obb',
       data="path/to/data.yaml",
       epochs=100,
       imgsz=640,
       batch=32
   )
   ```

2. **Server Training Setup**
   ```python
   model = YOLO('yolov8m-obb.pt')
   results = model.train(
       task='obb',
       data="/opt/dlami/nvme/chronoscope/yolo_broader_meter/data.yaml",
       epochs=200,
       imgsz=640,
       batch=32,
       augment=False,
       fliplr=0.0,
       flipud=0.0,
       mosaic=0.0,
       mixup=0.0,
       scale=0.0
   )
   ```

### Data Requirements
- **Dataset Structure**:
  ```
  dataset_root/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── val/
  │   ├── images/
  │   └── labels/
  ├── test/
  │   ├── images/
  │   └── labels/
  └── data.yaml
  ```

- **Data YAML Format**:
  ```yaml
  train: train/images
  val: val/images
  test: test/images
  nc: [number_of_classes]
  names: [class_names]
  ```

### Training Data Sources
The training data is stored in AWS S3 buckets:

1. **Individual Number Detection Dataset**:
   - Location: [AWS S3 Chronodata Bucket - Individual Numbers](https://ap-south-1.console.aws.amazon.com/s3/buckets/chronodata?region=ap-south-1&bucketType=general&prefix=dataset/training_data/yolo_indivisual/)

2. **Broad Meter Detection Dataset**:
   - Location: [AWS S3 Chronodata Bucket - Broader Meter](https://ap-south-1.console.aws.amazon.com/s3/buckets/chronodata?region=ap-south-1&bucketType=general&prefix=dataset/training_data/yolo_broader_meter/)

### Notes
- Supports both local and server-based training configurations
- Server configuration includes options to disable various augmentations
- Uses medium-sized YOLOv8 architecture (yolov8m-obb.pt)
- Training parameters can be adjusted based on available computational resources
- Includes optional validation and result saving functionality

## Script 5: Classification Data Creation (`9_classiification_data_creation.py`)

### Purpose
Script to organize and prepare image data for classification tasks by creating a structured directory hierarchy based on class labels. This script processes annotated images and organizes them into class-specific folders, making the dataset ready for training classification models.

### Configuration (`model_training_config.yaml`)
```yaml
image_classification:
  data_paths:
    source_dir: 'path/to/source/images'
    destination_dir: 'path/to/classification/dataset'
    annotations_file: 'path/to/annotations.csv'
  classes:
    - class_name_1
    - class_name_2
    # Add other classes as needed
```

### How It Works
1. **Configuration Loading**
   - Reads YAML configuration file
   - Sets up source and destination paths
   - Loads class definitions

2. **Directory Structure Creation**
   - Creates base destination directory
   - Creates subdirectories for each class
   - Handles spaces in class names by replacing with underscores

3. **Data Organization**
   - Reads annotations from CSV file
   - Copies images to respective class folders
   - Maintains class distribution statistics
   - Handles missing files with warnings

### Usage
```bash
# Run classification data organization script
python src/model_training/9_classiification_data_creation.py
```

### Data Flow
- **Input**:
  - Source directory containing original images
  - CSV file with annotations (format: image_name,class_label,...)
  - Configuration file with paths and class definitions
- **Output**:
  - Organized dataset structure:
    ```
    destination_dir/
    ├── class_1/
    │   └── [images for class 1]
    ├── class_2/
    │   └── [images for class 2]
    └── ...
    ```
  - Distribution statistics showing:
    - Count of images per class
    - Percentage distribution
    - Total image count

### File Formats
1. **Input CSV Format**
   ```text
   image_name,annotation,additional_fields
   image1.jpg,class_label_1,...
   image2.jpg,class_label_2,...
   ```

2. **Output Directory Structure**
   ```text
   destination_dir/
   ├── class_label_1/
   │   └── [corresponding images]
   ├── class_label_2/
   │   └── [corresponding images]
   └── ...
   ```

### Notes
- Automatically creates class directories if they don't exist
- Handles spaces in class names by converting to underscores
- Provides detailed distribution statistics for dataset analysis
- Reports warnings for missing source files
- Preserves original image files through copying
- Supports flexible CSV annotation format

## Script 6: Digit Color Classification Data Creation (`10_digit_classification_data_creation.py`)

### Purpose
Script to process meter images and extract the last digit, organizing them into separate directories based on their color (red or black). This script uses pre-trained models to detect digits, extract the last digit, and save it to the appropriate color-based directory for further classification tasks.

### Configuration (`model_training_config.yaml`)
```yaml
digit_color_classification:
  data_paths:
    annotations_file: 'dataset/last_digit_red_annotations.csv'
    output_dirs:
      red: 'dataset/data_cleaned/color_classification/red'
      black: 'dataset/data_cleaned/color_classification/black'

  models:
    bfm_classification: True  # Flag to indicate BFM classification model usage
    individual_numbers: True  # Flag to indicate individual numbers model usage

  processing:
    extract_last_digit: True  # Flag to indicate last digit extraction
    save_cropped_images: True  # Flag to indicate if cropped images should be saved
```

### How It Works
1. **Configuration Loading**
   - Reads YAML configuration file
   - Sets up paths for input annotations and output directories
   - Configures model usage and processing flags

2. **Image Processing Pipeline (`process_image`)**
   - Loads and validates input image
   - Performs BFM (Blurry/Foggy/Meter) classification
   - Detects and extracts individual digits
   - Isolates and saves the last digit
   - Handles various error cases gracefully

3. **Color-based Organization**
   - Reads color annotations from CSV file
   - Directs extracted digits to appropriate color directories
   - Maintains processing statistics
   - Reports success/failure counts

### Usage
```bash
# Run digit color classification script
python src/model_training/10_digit_classification_data_creation.py
```

### Data Flow
- **Input**:
  - CSV file containing:
    - Image paths
    - Color annotations (is_last_digit_red boolean)
  - Source images
  - Pre-trained models:
    - BFM classification model
    - Individual numbers detection model
- **Output**:
  - Organized dataset structure:
    ```
    data_cleaned/
    └── color_classification/
        ├── red/
        │   └── [extracted red digit images]
        └── black/
            └── [extracted black digit images]
    ```
  - Processing statistics:
    - Total images processed
    - Successfully extracted digits
    - Error counts and types

### Processing Steps
1. **Image Quality Check**
   - Uses BFM model to classify image quality
   - Only processes images classified as "good"
   - Filters out poor quality images early

2. **Digit Detection and Extraction**
   - Detects all digits in the meter
   - Identifies the last digit position
   - Extracts the digit using bounding box coordinates
   - Saves the cropped digit image

3. **Color-based Classification**
   - Determines digit color from annotations
   - Routes extracted digits to appropriate directory
   - Maintains original image names for traceability

### Notes
- Supports configurable model usage through YAML settings
- Handles missing files and processing errors gracefully
- Provides detailed processing statistics
- Creates output directories automatically if they don't exist
- Preserves original image names in extracted digits
- Uses pre-trained models for robust digit detection
- Integrates with existing BFM and digit recognition pipelines
- Suitable for creating color-based digit classification datasets

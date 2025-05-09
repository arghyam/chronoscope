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

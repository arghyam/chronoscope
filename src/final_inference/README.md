## Script 1:  Chronoscope Inference Utilities Documentation


The `inference_utils.py` module provides a comprehensive set of utilities for performing inference on meter images. It includes functionality for image classification, digit recognition, color detection, and image enhancement.

## Core Components

### 1. Configuration Management
```python
def load_config(config_path="src/final_inference/config.yaml"):
```
**Purpose**: Loads configuration settings from a YAML file for consistent parameter management across the inference pipeline.

**Returns**:
- Configuration dictionary containing model paths and parameters

### 2. Model Loading Functions

#### Bulk Flow Meter Classification
```python
def load_bfm_classification(model_path=None):
```
**Purpose**: Loads the FastAI model for classifying meter images as good or bad quality.

**Parameters**:
- `model_path`: Optional path to model file (defaults to config value)

**Returns**:
- Loaded FastAI learner object for classification

#### Individual Numbers Detection
```python
def load_individual_numbers_model(model_path=None):
```
**Purpose**: Loads the YOLO model for detecting individual digits in meter images.

**Parameters**:
- `model_path`: Optional path to model file (defaults to config value)

**Returns**:
- Loaded YOLO model object for digit detection

#### Color Classification
```python
def load_color_classification_model(model_path=None):
```
**Purpose**: Loads the FastAI model for classifying digit colors.

**Parameters**:
- `model_path`: Optional path to model file (defaults to config value)

**Returns**:
- Loaded FastAI learner object for color classification

### 3. Image Classification Functions

#### Bulk Flow Meter Image Classification
```python
def classify_bfm_image(image_path, model=None):
```
**Purpose**: Classifies a meter image as good or bad quality.

**Parameters**:
- `image_path`: Path to image file or numpy array
- `model`: Optional pre-loaded classification model

**Returns**:
```python
{
    'prediction': str,      # 'good' or 'bad'
    'confidence': float,    # Prediction confidence (0-1)
    'all_probs': list      # Probabilities for all classes
}
```

#### Color Classification
```python
def classify_color_image(image_path, model=None):
```
**Purpose**: Classifies the color of digits (red, black, or blue).

**Parameters**:
- `image_path`: Path to image file or numpy array
- `model`: Optional pre-loaded color classification model

**Returns**:
```python
{
    'prediction': str,      # Color class ('red', 'black', 'blue')
    'confidence': float,    # Prediction confidence (0-1)
    'all_probs': list      # Probabilities for all classes
}
```

### 4. Image Enhancement

```python
def enhance_image(image):
```
**Purpose**: Enhances meter images for improved digit recognition while preserving color information.

**Parameters**:
- `image`: Input image in BGR format (numpy array)

**Returns**:
- Enhanced image with improved contrast and sharpness

**Enhancement Steps**:
1. Converts image to HSV color space
2. Applies adaptive thresholding
3. Performs unsharp masking
4. Enhances contrast using CLAHE
5. Applies color boost

### 5. Digit Detection and Processing

#### Direct Meter Reading Recognition
```python
def direct_recognize_meter_reading(image_path, individual_numbers_model=None):
```
**Purpose**: Processes meter images to detect and recognize digit sequences.

**Parameters**:
- `image_path`: Path to input image
- `individual_numbers_model`: Optional pre-loaded YOLO model

**Returns**:
- Tuple containing:
  - Detected meter reading as string
  - Sorted bounding boxes
  - Sorted digit classes

**Processing Steps**:
1. Loads and enhances the image
2. Detects individual digits using YOLO model
3. Removes overlapping detections
4. Sorts digits left-to-right
5. Combines digits into final reading

### 6. Utility Functions

#### Box Processing
```python
def sort_boxes_by_position(boxes, classes):
```
**Purpose**: Sorts detected digit boxes from left to right for proper reading order.

```python
def calculate_iou(box1, box2):
```
**Purpose**: Calculates Intersection over Union between two polygon boxes.

```python
def remove_overlapping_boxes(boxes, classes, confidences, iou_threshold=0.5):
```
**Purpose**: Removes overlapping detections while keeping highest confidence ones.

#### Digit Extraction
```python
def extract_digit_image(image, box):
```
**Purpose**: Extracts individual digit images from their bounding boxes.

#### Color Analysis
```python
def is_last_digit_color_different_hsv(digit_crops, threshold=0.85):
```
**Purpose**: Analyzes if the last digit's color differs from others using HSV color space.

**Returns**:
```python
{
    'is_red': bool,        # True if color differs significantly
    'confidence': float    # Confidence in the color difference
}
```

## Usage Example
```python
# Example workflow for meter reading
test_image_path = CONFIG['test']['sample_image']

# Classify image quality
classification_result = classify_bfm_image(test_image_path)

# Process if image quality is good
if classification_result['prediction'].lower() == 'good':
    meter_reading = direct_recognize_meter_reading(test_image_path)
    print("Detected meter reading:", meter_reading)
else:
    print("Image classified as bad quality - skipping digit detection")
```

## Notes
- All models should be loaded before processing multiple images for better performance
- Image enhancement is automatically applied during digit recognition
- The module handles both file paths and numpy arrays as input
- Color analysis is performed in HSV space for better robustness
- Overlapping detection removal ensures accurate digit sequence recognition

## Script 2: Streamlit Web Interface Documentation

The `app.py` module provides a user-friendly web interface for the meter reading recognition system using Streamlit.

### 1. Application Setup

```python
@st.cache_resource
def load_models():
```
**Purpose**: Loads and caches all required models at application startup.

**Returns**:
- Dictionary containing:
  - `bfm_classification`: Bulk Flow Meter classification model
  - `individual_numbers`: Individual numbers detection model
  - `color_classification`: Color classification model

### 2. Main Interface Components

#### File Upload
- Accepts image files (jpg, jpeg, png)
- Displays uploaded image in the interface
- Provides clear instructions when no file is uploaded

#### Image Processing
```python
def process_uploaded_image(uploaded_file, models):
```
**Purpose**: Processes uploaded images through the complete recognition pipeline.

**Parameters**:
- `uploaded_file`: Uploaded file object from Streamlit
- `models`: Dictionary of pre-loaded models

**Returns**:
- Tuple containing:
  - `formatted_reading`: Recognized meter reading
  - `inference_time`: Processing time in seconds
  - `image`: Processed image
  - `classification_result`: Image quality classification
  - `quality_status`: Image quality status
  - `color_result`: Color classification of last digit

**Processing Steps**:
1. Saves uploaded file temporarily
2. Classifies image quality
3. If quality is good:
   - Recognizes meter reading
   - Extracts and classifies last digit color
   - Adjusts reading based on color (divides by 10 if red)
4. Cleans up temporary files
5. Measures inference time

### 3. User Interface Features

#### Layout
- Wide layout configuration
- Two-column design for image and results
- Clear heading and section separation

#### Results Display
- Processing time indicator
- Image quality status
- Recognized meter reading (if successful)
- Last digit color and confidence
- Error messages for bad quality images

#### User Guidance
- Clear instructions for usage
- File upload guidance
- Processing status indicators
- Success/error messages

### Usage Example
1. Launch the Streamlit app
2. Upload a meter image using the file uploader
3. Click "Recognize Meter Reading"
4. View results including:
   - Image quality assessment
   - Recognized meter reading
   - Last digit color classification
   - Processing time

### Notes
- Models are cached to improve performance across multiple uses
- Temporary files are properly managed and cleaned up
- User feedback is provided at each step of processing
- Interface is designed for intuitive use without technical knowledge
- Error handling ensures graceful failure in case of issues

## Script 3: GPU-Enabled Streamlit Web Interface

The `app_gpu.py` module extends the base Streamlit interface with GPU acceleration support for improved performance.

### 1. GPU Setup and Management

#### GPU Initialization
```python
def setup_gpu():
```
**Purpose**: Initializes and configures GPU for processing.

#### Memory Management
```python
def clear_gpu_memory():
```
**Purpose**: Manages GPU memory by clearing cache and running garbage collection.
- Frees up CUDA memory cache
- Runs Python garbage collection
- Called between processing steps to prevent memory issues

### 2. Enhanced Application Setup

```python
@st.cache_resource
def load_models():
```
**Purpose**: Loads and caches models with GPU support.

**Features**:
- Clears GPU memory before loading models
- Handles GPU-related exceptions
- Returns models dictionary or None if loading fails

### 3. GPU-Specific Interface Components

#### System Information Display
- GPU device name display in sidebar
- Total GPU memory information
- Fallback CPU notification if GPU unavailable

#### Resource Monitoring
- GPU memory usage tracking
- Memory allocation display after processing
- Processing time measurements

### 4. Enhanced Image Processing Pipeline

```python
def process_uploaded_image(uploaded_file, models):
```
**Purpose**: Processes images using GPU acceleration.

**GPU-Specific Features**:
- Memory clearing between processing stages
- Exception handling for GPU-related errors
- Resource cleanup after processing
- Memory usage monitoring

**Processing Steps**:
1. Clears GPU memory
2. Loads and processes image
3. Performs GPU-accelerated inference:
   - Image quality classification
   - Digit recognition
   - Color classification
4. Manages GPU resources
5. Cleans up temporary files
6. Reports GPU memory usage

### 5. Performance Considerations

#### Memory Management
- Automatic GPU memory clearing between operations
- Garbage collection after major processing steps
- Temporary file cleanup

#### Resource Optimization
- Cached model loading for repeated use
- Memory monitoring and reporting
- Exception handling for GPU-related issues

### Usage Example
1. Launch the GPU-enabled Streamlit app
2. Verify GPU availability in sidebar
3. Upload meter image
4. Monitor GPU memory usage during processing
5. View results with performance metrics

### Notes
- Requires CUDA-compatible GPU
- Automatically falls back to CPU if GPU unavailable
- Monitors and displays GPU resource usage
- Implements memory management best practices
- Provides enhanced error handling for GPU operations

## Script 4: Golden Test Data Annotation Tool

The `1_create_golden_testdata_annotation_streamlit.py` module provides a Streamlit-based interface for creating and validating golden test data for the meter reading system.

### 1. Core Functions

#### Image Loading
```python
def load_image(image_path):
```
**Purpose**: Loads and validates image files.
- Handles image loading exceptions
- Returns None for invalid images

#### CSV Management
```python
def load_csv(csv_path):
```
**Purpose**: Loads annotation data with proper data types.
- Preserves exact string representation of digit sequences
- Handles loading errors with user feedback

```python
def save_csv(df, csv_path):
```
**Purpose**: Saves annotation data with proper formatting.
- Maintains data integrity with proper quoting
- Provides error handling and feedback

### 2. Main Interface Components

#### Directory Structure
Expected folder structure:
```
base_folder/
├── images/
│   └── meter_images.jpg/png
└── csv_files/
    └── digit_sequences.csv
```

#### Data Management Features
- Progress tracking across sessions
- Missing image detection and handling
- Statistics tracking and display
- Data validation and preservation

### 3. User Interface Features

#### Navigation Controls
- Next/Previous image navigation
- Save and continue functionality
- Progress indicator
- Current position display

#### Annotation Tools
- Digit sequence editing
- Decimal point marking (red/black)
- Missing image handling
- Image preview display

#### Statistics Display
- Total images processed
- Missing image count
- Success rate calculation
- Real-time progress updates

### 4. Data Validation Features

#### Image Validation
- Checks for image existence
- Handles missing images gracefully
- Provides missing image statistics

#### Data Entry Validation
- Preserves exact digit sequences
- Tracks decimal point presence
- Maintains data integrity
- Prevents data loss

### Usage Example
1. Launch the annotation tool
2. Enter base folder path containing images and CSV
3. For each image:
   - Verify/edit digit sequence
   - Mark decimal point status
   - Handle missing images if needed
   - Save and continue

### Notes
- Maintains session state for continuous work
- Provides keyboard shortcuts for efficiency
- Includes comprehensive error handling
- Offers real-time statistics and progress tracking
- Ensures data integrity through proper type handling

### Best Practices
1. Regular saving of annotations
2. Consistent decimal point marking
3. Careful handling of missing images
4. Regular progress monitoring
5. Validation of entered data

This tool is essential for:
- Creating reliable test datasets
- Validating meter reading accuracy
- Training data preparation
- System performance evaluation

## Script 5: Model Performance Evaluation

The `2_check_performance_models.py` module provides functionality for evaluating the performance of the meter reading models against a golden dataset.

### 1. Core Components

#### Configuration and Model Loading
```python
def load_config():
```
**Purpose**: Loads configuration settings from YAML file specific to evaluation.

```python
def load_models(config):
```
**Purpose**: Loads all required models for evaluation:
- Bulk Flow Meter classification model
- Individual numbers detection model
- Color classification model

### 2. Evaluation Functions

#### Reading Comparison
```python
def compare_readings(true_sequence, recognized_reading):
```
**Purpose**: Compares true sequence with recognized reading.

**Parameters**:
- `true_sequence`: Ground truth reading from golden dataset
- `recognized_reading`: Model's recognized reading

**Returns**:
- `'correct'`: If readings match exactly
- `'wrong'`: If readings differ
- `'no_output'`: For cases of bad images, no digits, or errors

#### Image Processing
```python
def process_image(image_path, models):
```
**Purpose**: Processes a single image through the complete recognition pipeline.

**Parameters**:
- `image_path`: Path to the image file
- `models`: Dictionary containing loaded models

**Returns**:
- Tuple containing:
  - Formatted meter reading
  - Image quality prediction
  - Color prediction

**Processing Steps**:
1. Classifies image quality
2. If good quality:
   - Recognizes meter reading
   - Extracts and classifies last digit color
   - Adjusts reading based on color (divides by 10 if red)
3. Handles various error cases and edge conditions

### 3. Main Evaluation Process

```python
def evaluate_models():
```
**Purpose**: Evaluates model performance on the entire golden dataset.

**Features**:
- Processes all images in the golden dataset
- Compares predictions with ground truth
- Calculates performance metrics
- Saves detailed results to CSV

**Output Statistics**:
- Total number of images processed
- Correct readings count and percentage
- Wrong readings count and percentage
- No output cases count and percentage

### 4. Results Format

The evaluation results CSV includes:
- `image_name`: Name of the processed image
- `true_sequence`: Ground truth digit sequence
- `true_color`: Ground truth decimal point status
- `is_missing`: Whether image is missing
- `recognized_reading`: Model's reading output
- `quality_prediction`: Image quality classification
- `color_prediction`: Last digit color prediction
- `comparison_result`: Comparison outcome (correct/wrong/no_output)

### Usage Example
```python
# Run the evaluation
python src/final_inference/2_check_performance_models.py

# Results will be saved to the path specified in config.yaml
# under dataset.golden_dataset.csv_files.evaluation_results
```

### Notes
- Requires a properly structured golden dataset with images and annotations
- Handles missing images gracefully
- Provides comprehensive performance metrics
- Saves detailed results for further analysis
- Uses the same models as the inference pipeline

## Script 6: FastAPI REST Service

The `fastapi.py` module provides a RESTful API service for the meter reading recognition system using FastAPI framework.

### 1. API Configuration

#### Service Setup
```python
app = FastAPI(
    title="Bulk Flow Meter Reading API",
    description="API for detecting and reading bulk flow meter values from images",
    version="1.1.0"
)
```

**Features**:
- CORS middleware enabled for cross-origin requests
- Temporary file management for uploaded images
- Automatic model loading at startup
- Background task support for cleanup operations

### 2. Core Endpoints

#### Health Check
```http
GET /health
```
**Purpose**: Monitors API and model health status.

**Returns**:
```json
{
    "status": "healthy",
    "models_loaded": true,
    "timestamp": "2024-03-21T10:00:00"
}
```

#### Meter Reading Prediction
```http
POST /predict
```
**Purpose**: Processes uploaded meter images and returns readings with quality assessment.

**Parameters**:
- `file`: Image file (jpg, jpeg, png)
- `save_debug`: Boolean flag for debug image saving (optional)

**Returns**:
```json
{
    "reading": "string",
    "quality_status": "string",
    "quality_confidence": float,
    "last_digit_color": "string",
    "color_confidence": float,
    "processing_time": float,
    "request_id": "string",
    "timestamp": "string"
}
```

#### Model Information
```http
GET /model/info
```
**Purpose**: Provides information about the deployed models.

**Returns**: Details about quality classification, digit recognition, and color classification models.

### 3. Key Features

#### Model Management
- Automatic model loading at startup
- Global model state management
- Memory-efficient model sharing across requests

#### Image Processing Pipeline
1. Image quality assessment
2. Digit recognition (if quality is good)
3. Color classification of last digit
4. Reading formatting based on color

#### Error Handling
- Comprehensive exception handling
- Detailed error responses
- Automatic cleanup of temporary files

#### Performance Features
- Background task support for cleanup
- Efficient file handling
- Request ID tracking
- Processing time measurement

### Usage Example

Using cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@meter_image.jpg"
```

Using Python requests:
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("meter_image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()
```

### Running the Service
```bash
uvicorn src.final_inference.fastapi:app --host 0.0.0.0 --port 8000 --reload
```

### Notes
- Requires all models to be available at startup
- Handles concurrent requests efficiently
- Provides detailed API documentation via Swagger UI at `/docs`
- Implements proper cleanup of temporary files
- Includes comprehensive error handling and reporting

## Script 7: GPU-Accelerated FastAPI Service

The `fastapi_gpu.py` module provides a GPU-accelerated version of the REST API service, optimized for systems with CUDA-capable GPUs.

### 1. GPU-Specific Features

#### Hardware Acceleration
- Utilizes CUDA-enabled GPUs for faster inference
- Automatic GPU memory management
- Optimized model loading for GPU execution

#### Resource Management
- GPU memory cleanup between requests
- Efficient batch processing capabilities
- Automatic fallback to CPU if GPU is unavailable

### 2. Implementation Differences from CPU Version

#### Model Loading
```python
@app.on_event("startup")
async def load_models():
    """
    Loads models to GPU memory at startup
    """
    global models
    models['bfm_classification'] = load_bfm_classification()
    models['individual_numbers'] = load_individual_numbers_model()
    models['color_classification'] = load_color_classification_model()
```

#### Image Processing Pipeline
The GPU version maintains the same processing steps but executes them on GPU:
1. Image quality assessment (GPU-accelerated)
2. Digit recognition using YOLO (GPU-optimized)
3. Color classification (GPU-accelerated)
4. Reading formatting and post-processing

### 3. Performance Considerations

#### Memory Management
- Automatic GPU memory clearing between requests
- Efficient handling of concurrent requests
- Optimized memory usage for batch processing

#### Processing Optimizations
- Parallel processing of multiple images
- GPU-accelerated image preprocessing
- Reduced CPU-GPU data transfer overhead

### 4. Deployment Requirements

#### Hardware Requirements
- CUDA-capable NVIDIA GPU
- Sufficient GPU memory (recommended: 4GB+)
- Compatible CUDA drivers and runtime

#### Software Dependencies
- CUDA Toolkit
- cuDNN
- GPU-enabled PyTorch
- GPU-compatible OpenCV

### 5. Deployment Instructions

#### Environment Setup
```bash
# Install CUDA dependencies
conda install cudatoolkit cudnn

# Install GPU-enabled PyTorch
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
```

#### Running the Service
```bash
# Start the GPU-enabled API service
uvicorn src.final_inference.fastapi_gpu:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Performance Monitoring

#### GPU Resource Monitoring
- GPU memory usage tracking
- Processing time measurements
- Resource utilization metrics

#### Error Handling
- GPU-specific error detection
- Automatic fallback mechanisms
- Detailed error reporting

### 7. Best Practices

#### Optimization Tips
- Batch processing for multiple images
- Regular GPU memory cleanup
- Proper error handling for GPU-related issues

#### Resource Management
- Monitor GPU memory usage
- Implement proper cleanup procedures
- Handle concurrent requests efficiently

### Usage Examples

#### Basic Request
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("meter_image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()
```

#### Batch Processing
```python
import requests
from concurrent.futures import ThreadPoolExecutor

def process_image(image_path):
    with open(image_path, "rb") as f:
        response = requests.post(
            "http://localhost:8000/predict",
            files={"file": f}
        )
    return response.json()

# Process multiple images in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, image_paths))
```

### Notes
- GPU version provides significant speed improvements for batch processing
- Requires proper CUDA setup and compatible hardware
- Maintains API compatibility with CPU version
- Includes automatic fallback to CPU if GPU is unavailable
- Optimized for high-throughput scenarios

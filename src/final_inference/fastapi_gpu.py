import os
import tempfile
import time
import uuid
from datetime import datetime
from typing import Optional

import cv2
import torch
import uvicorn
from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.final_inference.inference_utilsgpu import classify_bfm_image
from src.final_inference.inference_utilsgpu import classify_color_image
from src.final_inference.inference_utilsgpu import direct_recognize_meter_reading
from src.final_inference.inference_utilsgpu import extract_digit_image
from src.final_inference.inference_utilsgpu import load_bfm_classification
from src.final_inference.inference_utilsgpu import load_color_classification_model
from src.final_inference.inference_utilsgpu import load_individual_numbers_model
from src.final_inference.inference_utilsgpu import setup_gpu

# Initialize GPU
device = setup_gpu()

app = FastAPI(
    title="Bulk Flow Meter Reading API",
    description="API for detecting and reading bulk flow meter values from images",
    version="1.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory for storing uploaded images if needed
TEMP_DIR = os.path.join(tempfile.gettempdir(), "meter_reading_api")
os.makedirs(TEMP_DIR, exist_ok=True)

# Models for API responses
class MeterReadingResponse(BaseModel):
    reading: str
    quality_status: str
    quality_confidence: float
    last_digit_color: str
    color_confidence: float
    processing_time: float
    request_id: str
    timestamp: str
    gpu_memory_used: Optional[float] = None

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Global variable to store loaded models
models = {
    'bfm_classification': None,
    'individual_numbers': None,
    'color_classification': None
}

# Add GPU memory management function
def clear_gpu_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

# Load models at startup
@app.on_event("startup")
async def load_models():
    global models
    try:
        clear_gpu_memory()
        models['bfm_classification'] = load_bfm_classification()
        models['individual_numbers'] = load_individual_numbers_model()
        models['color_classification'] = load_color_classification_model()
    except Exception as e:
        print(f"Error loading models: {str(e)}")

def cleanup_temp_file(file_path: str):
    """Remove temporary file after processing"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error removing temp file {file_path}: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API info"""
    gpu_info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if torch.cuda.is_available() else None
    }

    return {
        "name": "Bulk Flow Meter Reading API",
        "version": "1.1.0",
        "gpu_info": gpu_info,
        "endpoints": [
            {"path": "/", "method": "GET", "description": "This information"},
            {"path": "/health", "method": "GET", "description": "API health check"},
            {"path": "/predict", "method": "POST", "description": "Submit an image for meter reading detection"},
            {"path": "/model/info", "method": "GET", "description": "Get information about the models"}
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = {
            "total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
            "allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
        }

    return {
        "status": "healthy",
        "models_loaded": all(model is not None for model in models.values()),
        "gpu_status": {
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory": gpu_memory
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=MeterReadingResponse)
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    save_debug: bool = False
):
    """
    Process an uploaded image and return the detected meter reading with quality assessment
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    initial_gpu_memory = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0

    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG image formats are supported"
        )

    try:
        # Clear GPU memory before processing
        clear_gpu_memory()

        # Create temporary file
        temp_file_path = os.path.join(TEMP_DIR, f"{request_id}_{file.filename}")
        content = await file.read()

        with open(temp_file_path, "wb") as f:
            f.write(content)

        background_tasks.add_task(cleanup_temp_file, temp_file_path)

        # Process image quality
        classification_result = classify_bfm_image(temp_file_path, model=models['bfm_classification'])
        quality_status = classification_result['prediction'].lower()

        if quality_status == 'good':
            # Perform digit recognition
            meter_reading, sorted_boxes, sorted_classes = direct_recognize_meter_reading(
                temp_file_path,
                models['individual_numbers']
            )

            # Process color classification if digits were detected
            if sorted_boxes and len(sorted_boxes) > 0:
                image = cv2.imread(temp_file_path)
                last_box = sorted_boxes[-1]
                last_digit_image = extract_digit_image(image, last_box)

                color_result = classify_color_image(
                    last_digit_image,
                    model=models['color_classification']
                )

                try:
                    original_length = len(meter_reading)
                    numeric_reading = float(meter_reading)

                    if color_result['prediction'].lower() == 'red':
                        numeric_reading = numeric_reading / 10
                        formatted_reading = f"{numeric_reading:.1f}".zfill(original_length + 2)
                    else:
                        formatted_reading = str(int(numeric_reading)).zfill(original_length)
                except ValueError:
                    formatted_reading = meter_reading
            else:
                formatted_reading = "No digits detected"
                color_result = {"prediction": "unknown", "confidence": 0.0}
        else:
            formatted_reading = "Image quality too poor for recognition"
            color_result = {"prediction": "unknown", "confidence": 0.0}

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Calculate GPU memory used
        gpu_memory_used = None
        if torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated(0)
            gpu_memory_used = (final_gpu_memory - initial_gpu_memory) / 1024**2  # Convert to MB

        return MeterReadingResponse(
            reading=formatted_reading,
            quality_status=quality_status,
            quality_confidence=classification_result['confidence'],
            last_digit_color=color_result['prediction'],
            color_confidence=color_result['confidence'],
            processing_time=processing_time,
            request_id=request_id,
            timestamp=datetime.now().strftime("%d/%m/%Y %I:%M %p"),
            gpu_memory_used=gpu_memory_used
        )

    except Exception as e:
        print(f"Error processing request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
    finally:
        # Clear GPU memory after processing
        clear_gpu_memory()

@app.get("/model/info")
async def model_info():
    """Get information about the models used for inference"""
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "device": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        }

    return {
        "models": {
            "quality_classification": {
                "name": "BFM Quality Classification Model",
                "classes": ["good", "bad"]
            },
            "digit_recognition": {
                "name": "YOLO Oriented Bounding Box (OBB) Individual Digits",
                "supported_digits": list(range(10)),
                "min_confidence_threshold": 0.3
            },
            "color_classification": {
                "name": "Last Digit Color Classification Model",
                "classes": ["red", "black"]
            }
        },
        "gpu_info": gpu_info
    }

if __name__ == "__main__":
    uvicorn.run("fastapi_gpu:app", host="0.0.0.0", port=8000, reload=True)


#to run
#uvicorn src.final_inference.fastapi_gpu:app --host 0.0.0.0 --port 8000 --reload

import os
import tempfile
import time
import uuid
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference.inference1 import direct_recognize_meter_reading
# Import inference function from inference1.py

app = FastAPI(
    title="Bulk Flow Meter Reading API",
    description="API for detecting and reading bulk flow meter values from images",
    version="1.0.0"
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
    confidence: float
    processing_time: float  # Processing time in milliseconds
    request_id: str
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Cleanup function for background tasks
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
    return {
        "name": "Bulk Flow Meter Reading API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "This information"},
            {"path": "/health", "method": "GET", "description": "API health check"},
            {"path": "/predict", "method": "POST", "description": "Submit an image for meter reading detection"}
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=MeterReadingResponse)
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                  save_debug: bool = False):
    """
    Process an uploaded image and return the detected meter reading

    - **file**: Image file to process (jpg, jpeg, png)
    - **save_debug**: Whether to save debug images (default: False)
    """
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG image formats are supported"
        )

    try:
        # Read image content
        content = await file.read()

        # Method 1: Process in memory
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. Please check the image format."
            )

        # Get a unique temporary file path for this request
        temp_file_path = os.path.join(TEMP_DIR, f"{request_id}_{file.filename}")

        # Save the image temporarily for processing
        with open(temp_file_path, "wb") as f:
            f.write(content)

        # Schedule cleanup of the temporary file
        background_tasks.add_task(cleanup_temp_file, temp_file_path)

        # Debug directory path if save_debug is True
        debug_dir = None
        if save_debug:
            debug_dir = os.path.join(TEMP_DIR, f"debug_{request_id}")
            os.makedirs(debug_dir, exist_ok=True)

        # Process the image using the direct recognition function
        meter_reading = direct_recognize_meter_reading(
            temp_file_path,
            save_debug_images=save_debug,
            debug_dir=debug_dir if debug_dir else "debug_images"
        )

        # Calculate processing time in milliseconds
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Prepare response
        response = MeterReadingResponse(
            reading=meter_reading,
            confidence=0.95,  # Confidence could be calculated from detection confidences
            processing_time=processing_time,
            request_id=request_id,
            timestamp=datetime.now().strftime("%d/%m/%Y %I:%M %p")
        )

        return response

    except Exception as e:
        # Log the error
        print(f"Error processing request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/model/info")
async def model_info():
    """Get information about the model used for inference"""
    return {
        "model_name": "YOLO Oriented Bounding Box (OBB) Individual Digits",
        "model_path": "runs_indi_full_10k/obb/train/weights/best.pt",
        "supported_digits": list(range(10)),  # 0-9
        "min_confidence_threshold": 0.3,
    }

if __name__ == "__main__":
    # Run the FastAPI app using uvicorn
    uvicorn.run("fastapi:app", host="0.0.0.0", port=8000, reload=True)

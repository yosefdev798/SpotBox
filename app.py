"""
SpotBox - FastAPI Object Detection Service
Exposes YOLOv8 detection via REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import requests
import numpy as np
from typing import List
import os
from datetime import datetime

app = FastAPI(
    title="SpotBox API",
    description="Object Detection API using YOLOv8",
    version="1.0.0"
)

# Load model once when server starts
MODEL_PATH = "yolov8n.pt"
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ Model not found at {MODEL_PATH}")
    print("📥 Downloading YOLOv8n model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded!")

class DetectRequest(BaseModel):
    image_url: str
    conf_threshold: float = 0.5

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class DetectResponse(BaseModel):
    detections: List[Detection]
    count: int
    timestamp: str

@app.get("/")
def root():
    return {
        "message": "SpotBox API is running",
        "endpoints": {
            "POST /detect/": "Detect objects in image from URL",
            "GET /health": "Check API health"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_PATH}

@app.post("/detect/", response_model=DetectResponse)
async def detect_objects(request: DetectRequest):
    """
    Detect objects in an image provided via URL.
    
    Example:
    ```json
    {
        "image_url": "https://example.com/test.jpg",
        "conf_threshold": 0.5
    }
    """
    try:
        # 1. Download the image from the URL
        response = requests.get(request.image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image from URL")

        # 2. Convert the downloaded bytes to an OpenCV image format
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # 3. Run YOLO detection
        results = model(img, conf=request.conf_threshold)

        # 4. Format the results into our Response model
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append(Detection(
                    class_name=model.names[int(box.cls)],
                    confidence=float(box.conf),
                    bbox=[float(x) for x in box.xyxy[0]]
                ))

        # 5. Return the final response
        return DetectResponse(
            detections=detections,
            count=len(detections),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        print(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
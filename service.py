import os
import time
import json
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Import your existing classes
from Analyser_Batch import DetectionConfig, BatchDefectDetector
from TextExtractor.ExtrctInfo import TextExtractor
from Reporter.ExcelReporter import ExcelReporter
from utils.logger import ProgressLogger

# Global variables for models
detector = None
text_extractor = None

PROGRESS_PATH = os.path.join(os.path.dirname(__file__), "progress_log.json")

class AnalyzeRequest(BaseModel):
    video_path: str

class AnalyzeResponse(BaseModel):
    ok: bool
    output_dir: str
    message: Optional[str] = None

def set_progress(stage: str, progress: float, extra: dict = None):
    """Update progress_log.json for UI compatibility"""
    try:
        data = {
            "current_stage": stage,
            "progress": progress,
            "stages": {
                "initialization": {"weight": 5, "progress": 100 if stage != "initialization" else progress},
                "frame extraction": {"weight": 10, "progress": 0},
                "Ai detection": {"weight": 40, "progress": 0},
                "text extraction": {"weight": 30, "progress": 0},
                "excel reporting": {"weight": 15, "progress": 0},
            }
        }
        if extra:
            if stage in data["stages"]:
                data["stages"][stage].update(extra)

        with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error updating progress: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events"""
    global detector, text_extractor

    # Startup
    try:
        print("Loading AI Sewer Pipe Analyzer Service...")
        set_progress("initialization", 10.0, {"status": "Loading models..."})

        # Get paths
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, 'Model', 'Model V.2.8.0', 'model_final.pth')
        config_path = os.path.join(base_dir, 'Model', 'Model V.2.8.0', 'mask_rcnn_X_101_32x8d_FPN_3x.yaml')

        # Validate paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print(f"Loading models from {model_path}")

        # Configure detection
        config = DetectionConfig(
            class_names=["Crack", "Obstacle", "Deposits", "Deformed",
                        "Broken", "Joint Displaced", "Surface Damage", "Root"],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                   [255, 0, 255], [0, 255, 255], [128, 128, 128], [128, 0, 128]],
            model_path=model_path,
            config_path=config_path,
            batch_size=16,
        )

        set_progress("initialization", 30.0, {"status": "Loading OCR models..."})

        # Load OCR models (with caching for faster subsequent runs)
        text_extractor = TextExtractor(lazy_load=True, use_cache=True)

        set_progress("initialization", 70.0, {"status": "Loading detection model..."})

        # Load detection model
        detector = BatchDefectDetector(config, text_extractor)

        set_progress("initialization", 100.0, {"status": "Ready"})

        print("All models loaded successfully!")
        print(f"OCR cache size: {TextExtractor.get_cache_size()} instances")

    except Exception as e:
        error_msg = f"Failed to load models: {str(e)}"
        print(f"Error: {error_msg}")
        set_progress("initialization", 0.0, {"status": error_msg})
        raise

    yield

    # Shutdown
    print("Shutting down service...")

# Create FastAPI app with lifespan events
app = FastAPI(title="AI Sewer Pipe Analyzer Service", lifespan=lifespan)

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "models_loaded": detector is not None and text_extractor is not None,
        "uptime": "Ready to process videos"
    }

@app.get("/status")
def status():
    """Get current service status"""
    return {
        "models_loaded": detector is not None and text_extractor is not None,
        "ocr_cache_size": TextExtractor.get_cache_size() if text_extractor else 0,
        "progress_file": PROGRESS_PATH if os.path.exists(PROGRESS_PATH) else None
    }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_video(request: AnalyzeRequest):
    """Analyze a video file"""
    global detector

    if not detector or not text_extractor:
        raise HTTPException(status_code=503, detail="Models not loaded. Service not ready.")

    video_path = request.video_path

    # Validate video file
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=400, detail=f"Video file not found: {video_path}")

    try:
        print(f"Processing video: {video_path}")

        # Create output directory
        output_dir = os.path.join(
            os.path.dirname(video_path),
            os.path.splitext(os.path.basename(video_path))[0] + "_output"
        )
        os.makedirs(output_dir, exist_ok=True)

        # Reset detector for new video
        detector.reset_for_new_video()

        # Process video
        frames_processed = detector.process_video(
            video_path,
            output_dir,
            batch_size=16  # Use fixed batch size
        )

        # Generate Excel report
        reporter = ExcelReporter(
            input_path=output_dir,
            excelOutPutName="Condition-Details.xlsx",
            progress_logger=detector.progress_logger
        )
        reporter.generate_report()

        print(f"✅ Video processed successfully: {output_dir}")

        return AnalyzeResponse(
            ok=True,
            output_dir=output_dir,
            message=f"Processed {frames_processed} frames successfully"
        )

    except Exception as e:
        error_msg = f"Error processing video: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/clear_cache")
def clear_cache():
    """Clear OCR model cache"""
    try:
        TextExtractor.clear_model_cache()
        return {"ok": True, "message": "Cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache_info")
def cache_info():
    """Get cache information"""
    return {
        "cache_size": TextExtractor.get_cache_size(),
        "models_loaded": detector is not None,
        "text_extractor_ready": text_extractor is not None
    }

if __name__ == "__main__":
    print("Starting AI Sewer Pipe Analyzer Service...")
    print("Service will be available at: http://127.0.0.1:8766")
    print("Endpoints:")
    print("   GET  /           - Health check")
    print("   GET  /status     - Service status")
    print("   GET  /cache_info - Cache information")
    print("   POST /analyze    - Analyze video")
    print("   POST /clear_cache - Clear model cache")

    # Run with single worker to keep models in memory
    # Use port 8766 to avoid conflicts (8765 might be in use)
    uvicorn.run(app, host="127.0.0.1", port=8766, workers=1)

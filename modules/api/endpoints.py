"""API endpoints for Ötüken3D."""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import asyncio
from typing import List, Optional, Dict, Any
import json
import time
from uuid import uuid4

from ..core.logger import setup_logger
from ..core.exceptions import ModelError, ProcessingError
from ..model3d_integration.text_to_shape import TextToShape
from ..model3d_integration.image_to_shape import ImageToShape
from ..core.constants import SUPPORTED_FORMATS

logger = setup_logger(__name__)

# API Models
class TextToShapeRequest(BaseModel):
    """Request model for text-to-shape endpoint."""
    text: str = Field(..., description="Text description of the desired 3D shape")
    style: Optional[List[str]] = Field(default=["Geleneksel"], description="Style options")
    format: str = Field(default="obj", description="Output format")
    resolution: Optional[int] = Field(default=128, description="Voxel resolution")
    use_diffusion: Optional[bool] = Field(default=False, description="Use diffusion refinement")

class ImageToShapeRequest(BaseModel):
    """Request model for image-to-shape endpoint."""
    format: str = Field(default="obj", description="Output format")
    resolution: Optional[int] = Field(default=128, description="Voxel resolution")

class ModelResponse(BaseModel):
    """Response model for model endpoints."""
    task_id: str = Field(..., description="Task ID for tracking progress")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Status message")

class TaskStatus(BaseModel):
    """Task status response model."""
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    progress: float = Field(..., description="Task progress (0-100)")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task result")
    error: Optional[str] = Field(default=None, description="Error message if failed")

# API Setup
app = FastAPI(
    title="Ötüken3D API",
    description="API for text-to-3D and image-to-3D model generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Task management
tasks = {}

class TaskManager:
    """Task management utility."""
    
    @staticmethod
    def create_task() -> str:
        """Create a new task."""
        task_id = str(uuid4())
        tasks[task_id] = {
            "status": "pending",
            "progress": 0.0,
            "result": None,
            "error": None
        }
        return task_id
    
    @staticmethod
    def update_task(task_id: str, **kwargs):
        """Update task status."""
        if task_id in tasks:
            tasks[task_id].update(kwargs)
    
    @staticmethod
    def get_task(task_id: str) -> Dict[str, Any]:
        """Get task status."""
        return tasks.get(task_id, {
            "status": "not_found",
            "progress": 0.0,
            "result": None,
            "error": "Task not found"
        })

# Model instances
text_to_shape = None
image_to_shape = None

def load_models():
    """Load models on startup."""
    global text_to_shape, image_to_shape
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if text_to_shape is None:
            text_to_shape = TextToShape(device=device)
            logger.info("Text-to-Shape model loaded")
        
        if image_to_shape is None:
            image_to_shape = ImageToShape(device=device)
            logger.info("Image-to-Shape model loaded")
            
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    load_models()

# Endpoints
@app.post("/api/v1/text-to-shape", response_model=ModelResponse)
async def text_to_shape_endpoint(
    request: TextToShapeRequest,
    background_tasks: BackgroundTasks
):
    """Generate 3D model from text description."""
    try:
        # Validate format
        if request.format.lower() not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Supported formats: {SUPPORTED_FORMATS}"
            )
        
        # Create task
        task_id = TaskManager.create_task()
        
        # Add task to background
        background_tasks.add_task(
            process_text_to_shape,
            task_id,
            request
        )
        
        return ModelResponse(
            task_id=task_id,
            status="pending",
            message="Task created successfully"
        )
        
    except Exception as e:
        logger.error(f"Text-to-Shape request failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/api/v1/image-to-shape", response_model=ModelResponse)
async def image_to_shape_endpoint(
    images: List[UploadFile] = File(...),
    format: str = "obj",
    resolution: int = 128,
    background_tasks: BackgroundTasks
):
    """Generate 3D model from images."""
    try:
        # Validate format
        if format.lower() not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Supported formats: {SUPPORTED_FORMATS}"
            )
        
        # Validate images
        if not images:
            raise HTTPException(
                status_code=400,
                detail="No images provided"
            )
        
        # Create task
        task_id = TaskManager.create_task()
        
        # Create temporary directory for images
        temp_dir = Path(tempfile.mkdtemp())
        image_paths = []
        
        # Save uploaded images
        for img in images:
            img_path = temp_dir / img.filename
            with open(img_path, "wb") as f:
                shutil.copyfileobj(img.file, f)
            image_paths.append(str(img_path))
        
        # Add task to background
        background_tasks.add_task(
            process_image_to_shape,
            task_id,
            image_paths,
            format,
            resolution,
            temp_dir
        )
        
        return ModelResponse(
            task_id=task_id,
            status="pending",
            message="Task created successfully"
        )
        
    except Exception as e:
        logger.error(f"Image-to-Shape request failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/api/v1/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get task status."""
    task = TaskManager.get_task(task_id)
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        result=task["result"],
        error=task["error"]
    )

async def process_text_to_shape(task_id: str, request: TextToShapeRequest):
    """Process text-to-shape request."""
    try:
        TaskManager.update_task(task_id, status="processing", progress=0.0)
        
        # Generate 3D model
        full_text = f"{request.text} (Stil: {', '.join(request.style)})"
        mesh = text_to_shape.predict(
            full_text,
            return_mesh=True
        )
        
        TaskManager.update_task(task_id, progress=50.0)
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(
            suffix=f".{request.format}",
            delete=False
        ) as tmp:
            mesh.export(tmp.name)
            
            # Create download URL
            result = {
                "format": request.format,
                "file_path": tmp.name
            }
            
            TaskManager.update_task(
                task_id,
                status="completed",
                progress=100.0,
                result=result
            )
            
    except Exception as e:
        logger.error(f"Text-to-Shape processing failed: {str(e)}")
        TaskManager.update_task(
            task_id,
            status="failed",
            progress=0.0,
            error=str(e)
        )

async def process_image_to_shape(
    task_id: str,
    image_paths: List[str],
    format: str,
    resolution: int,
    temp_dir: Path
):
    """Process image-to-shape request."""
    try:
        TaskManager.update_task(task_id, status="processing", progress=0.0)
        
        # Generate 3D model
        mesh = image_to_shape.predict(
            image_paths,
            return_mesh=True
        )
        
        TaskManager.update_task(task_id, progress=50.0)
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(
            suffix=f".{format}",
            delete=False
        ) as tmp:
            mesh.export(tmp.name)
            
            # Create download URL
            result = {
                "format": format,
                "file_path": tmp.name
            }
            
            TaskManager.update_task(
                task_id,
                status="completed",
                progress=100.0,
                result=result
            )
            
    except Exception as e:
        logger.error(f"Image-to-Shape processing failed: {str(e)}")
        TaskManager.update_task(
            task_id,
            status="failed",
            progress=0.0,
            error=str(e)
        )
    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir)

@app.get("/api/v1/download/{task_id}")
async def download_model(task_id: str):
    """Download generated model."""
    task = TaskManager.get_task(task_id)
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Task not completed"
        )
    
    if not task["result"]:
        raise HTTPException(
            status_code=404,
            detail="Result not found"
        )
    
    file_path = task["result"]["file_path"]
    if not Path(file_path).exists():
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )
    
    return FileResponse(
        file_path,
        filename=f"model.{task['result']['format']}",
        media_type="application/octet-stream"
    ) 
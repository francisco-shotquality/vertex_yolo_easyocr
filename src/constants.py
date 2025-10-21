"""
constants.py

Defines configuration constants for the inference server. Environment
variables are loaded from a `.env` file if present using python-dotenv.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file at project root (if exists)
# This call does nothing if the file is absent.
load_dotenv()

# HTTP port used by Vertex AI custom container. Default matches Vertex AI spec.
AIP_HTTP_PORT = int(os.getenv("AIP_HTTP_PORT", "8080"))

# API routes (health and prediction) for the FastAPI app.
AIP_HEALTH_ROUTE = os.getenv("AIP_HEALTH_ROUTE", "/health")
AIP_PREDICT_ROUTE = os.getenv("AIP_PREDICT_ROUTE", "/predict")

# Path to the YOLO weights file. You can override via environment variable.
YOLO_WEIGHTS_PATH = os.getenv("YOLO_WEIGHTS_PATH", str(Path("models") / "best.pt"))

# Languages used by EasyOCR (comma-separated). Default to English only.
OCR_LANGUAGES = [lang.strip() for lang in os.getenv("OCR_LANGUAGES", "en").split(",") if lang.strip()]

# Minimum detection confidence threshold for YOLO. Default 0.25.
DEFAULT_DETECTION_CONFIDENCE = float(os.getenv("DEFAULT_DETECTION_CONFIDENCE", "0.25"))

'''Main entrypoint for the Vertex AI custom container.

This module defines a FastAPI application that loads a YOLO model for
region‑of‑interest (ROI) detection and an EasyOCR reader for optical
character recognition.  When deployed to Vertex AI as a custom
container, the server exposes a health check and a prediction
endpoint.  The prediction endpoint accepts a batch of images encoded
as base64 strings and returns, for each image, the detected bounding
boxes and the text recognized within those boxes.

The implementation follows the general recommendations for serving
models on Vertex AI.  A persistent model instance is created at
import time, and the FastAPI application uses Pydantic models to
validate input and output.  Logging is handled through the `loguru`
package, which sends messages to standard output as recommended by
documentation.'''

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
from loguru import logger
from .constants import (
    AIP_HTTP_PORT,
    AIP_HEALTH_ROUTE,
    AIP_PREDICT_ROUTE,
    YOLO_WEIGHTS_PATH,
    OCR_LANGUAGES,
    DEFAULT_DETECTION_CONFIDENCE,
)


try:
    # Import Ultralytics lazily to avoid overhead if the module is not
    # available.  Ultralytics provides a simple API for running YOLO
    # inference.
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        'Ultralytics is required but not installed.  Ensure that the '
        '`ultralytics` package is listed in your pyproject.toml dependencies.'
    ) from exc

try:
    import easyocr
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        'EasyOCR is required but not installed.  Ensure that the '
        '`easyocr` package is listed in your pyproject.toml dependencies.'
    ) from exc


# ---------------------------------------------------------------------------
# Environment variables and configuration
# ---------------------------------------------------------------------------

# Vertex AI populates the following environment variables for custom
# containers.  These variables are used to determine the HTTP port and
# endpoint paths.  We provide sensible defaults to allow local testing.
AIP_HTTP_PORT: int = int(os.getenv('AIP_HTTP_PORT', '8080'))
AIP_HEALTH_ROUTE: str = os.getenv('AIP_HEALTH_ROUTE', '/health')
AIP_PREDICT_ROUTE: str = os.getenv('AIP_PREDICT_ROUTE', '/predict')

# The path to the YOLO model weights can be supplied via an environment
# variable.  The default points to a file under the `models` directory.
YOLO_WEIGHTS_PATH: str = os.getenv(
    'YOLO_WEIGHTS_PATH', os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8n.pt')
)

# Languages for EasyOCR.  Use comma‑separated ISO codes.  Default to
# English only.  See https://www.jaided.ai/easyocr/ for supported codes.
OCR_LANGS: List[str] = os.getenv('OCR_LANGS', 'en').split(',')

# Minimum confidence for YOLO detections.  You may override this via
# `parameters` in the prediction request.
DEFAULT_CONFIDENCE: float = float(os.getenv('DEFAULT_CONFIDENCE', '0.5'))

# Initialize the FastAPI app.
app = FastAPI()

# Flags to indicate whether models have been loaded successfully.
_models_ready: bool = False
_yolo_model: YOLO | None = None
_ocr_reader: easyocr.Reader | None = None


def _initialize_models() -> None:
    '''Load the YOLO and EasyOCR models.

    This function is called once at import time to avoid reloading
    models for every request.  It sets the global flags and logs any
    errors encountered.  Loading may take a few seconds depending on
    model size and GPU availability.
    '''
    global _models_ready, _yolo_model, _ocr_reader
    if _models_ready:
        return
    try:
        logger.info('Loading YOLO model from {}', YOLO_WEIGHTS_PATH)
        _yolo_model = YOLO(YOLO_WEIGHTS_PATH)
        logger.info('YOLO model loaded successfully')
    except Exception as e:
        logger.error('Failed to load YOLO model: {}', e)
        _yolo_model = None
    try:
        logger.info('Initializing EasyOCR reader for languages: {}', OCR_LANGS)
        _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=True)
        logger.info('EasyOCR reader initialized successfully')
    except Exception as e:
        logger.error('Failed to initialize EasyOCR reader: {}', e)
        _ocr_reader = None
    _models_ready = _yolo_model is not None and _ocr_reader is not None


# Initialize models at module import time.  Vertex AI requires that
# models are ready when the first request arrives【612638540015317†L273-L297】.
_initialize_models()


def _decode_image(b64_data: str) -> Image.Image:
    '''Decode a base64‑encoded image into a PIL Image.

    Args:
        b64_data: Base64‑encoded image string.

    Returns:
        A PIL Image in RGB mode.

    Raises:
        HTTPException: If the image cannot be decoded or is in an
            unsupported format.
    '''
    try:
        binary_data = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(binary_data)).convert('RGB')
        return img
    except Exception as e:
        logger.error('Failed to decode image: {}', e)
        raise HTTPException(status_code=400, detail='Invalid image data') from e


def _run_detection(image: Image.Image, confidence: float) -> List[Dict[str, Any]]:
    '''Run YOLO detection on a single image and return bounding boxes.

    Args:
        image: A PIL Image in RGB mode.
        confidence: Minimum confidence threshold for detections.

    Returns:
        A list of dictionaries, each containing bounding box coordinates,
        class name and confidence.
    '''
    if _yolo_model is None:
        raise RuntimeError('YOLO model is not loaded')
    # Use the Ultralytics API to predict.  We set verbose=False to
    # suppress console output and save=False to avoid writing files.
    try:
        results = _yolo_model.predict(
            source=image,
            conf=confidence,
            save=False,
            verbose=False,
        )
    except Exception as e:
        logger.error('Error during YOLO inference: {}', e)
        raise
    # Extract bounding boxes from the first result (one image)
    detections: List[Dict[str, Any]] = []
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                detections.append(
                    {
                        'xmin': float(xyxy[i][0]),
                        'ymin': float(xyxy[i][1]),
                        'xmax': float(xyxy[i][2]),
                        'ymax': float(xyxy[i][3]),
                        'confidence': float(confs[i]),
                        'class_id': int(classes[i]),
                        'class_name': _yolo_model.names.get(int(classes[i]), f'class_{int(classes[i])}'),
                    }
                )
    return detections


def _run_ocr(cropped: Image.Image) -> List[Dict[str, Any]]:
    '''Run EasyOCR on a cropped region and return recognized text.

    Args:
        cropped: A cropped PIL Image containing potential text.

    Returns:
        A list of dictionaries with text and confidence.  Bounding box
        coordinates returned by EasyOCR are relative to the cropped
        region; they are not propagated back to the original image.
    '''
    if _ocr_reader is None:
        raise RuntimeError('EasyOCR reader is not loaded')
    try:
        result = _ocr_reader.readtext(np.array(cropped), detail=1)
    except Exception as e:
        logger.error('Error during OCR: {}', e)
        raise
    ocr_results: List[Dict[str, Any]] = []
    for bbox, text, conf in result:
        ocr_results.append(
            {
                'text': text,
                'confidence': float(conf),
                'bbox': [
                    [float(pt[0]), float(pt[1])] for pt in bbox
                ],
            }
        )
    return ocr_results


class Instance(BaseModel):
    '''Schema for a single prediction instance.

    Vertex AI sends instances as dictionaries, so this model
    primarily documents the expected fields.  The `image` field
    contains a base64‑encoded string.  Additional fields can be
    included as needed.
    '''
    image: str = Field(..., description='Base64 encoded image data')


class PredictionRequest(BaseModel):
    '''Request schema for the predict endpoint.'''
    instances: List[Instance]
    parameters: Optional[Dict[str, Any]] = None


class DetectionResult(BaseModel):
    '''Response structure for a single detection.'''
    bbox: Dict[str, float]
    class_name: str
    confidence: float
    ocr: Optional[List[Dict[str, Any]]] = None


class PredictionResponse(BaseModel):
    '''Response schema for the predict endpoint.'''
    predictions: List[Dict[str, Any]]


@app.get(AIP_HEALTH_ROUTE, status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    '''Health check endpoint required by Vertex AI.

    Returns 503 if models have not finished loading.
    '''
    if not _models_ready:
        raise HTTPException(status_code=503, detail='Models not ready')
    return {'status': 'healthy'}


@app.post(AIP_PREDICT_ROUTE, response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    '''Prediction endpoint for batch inference.

    The request should contain a list of instances where each
    instance includes a base64 encoded image.  Optional parameters
    include the detection confidence threshold (`confidence`) and
    whether to run OCR on the detected regions (`run_ocr`).

    Returns a list of predictions corresponding to the input
    instances.  Each prediction includes the detected bounding boxes
    with class names and, if OCR was requested, recognized text
    inside those boxes.
    '''
    if not _models_ready:
        raise HTTPException(status_code=503, detail='Models not ready')
    params = request.parameters or {}
    confidence: float = float(params.get('confidence', DEFAULT_DETECTION_CONFIDENCE))
    run_ocr: bool = bool(params.get('run_ocr', True))

    predictions: List[Dict[str, Any]] = []
    for idx, instance in enumerate(request.instances):
        # Decode the image
        image = _decode_image(instance.image)
        # Run detection
        detections = _run_detection(image, confidence=confidence)
        prediction: Dict[str, Any] = {
            'detections': [],
            'detection_count': len(detections),
        }
        # For each detection, optionally run OCR on the cropped region
        for det in detections:
            xmin, ymin, xmax, ymax = (
                det['xmin'],
                det['ymin'],
                det['xmax'],
                det['ymax'],
            )
            bbox_dict = {
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
            }
            det_result = {
                'bbox': bbox_dict,
                'class_name': det['class_name'],
                'confidence': det['confidence'],
            }
            if run_ocr:
                # Crop the ROI and run OCR
                left = max(int(xmin), 0)
                upper = max(int(ymin), 0)
                right = min(int(xmax), image.width)
                lower = min(int(ymax), image.height)
                cropped = image.crop((left, upper, right, lower))
                try:
                    ocr_results = _run_ocr(cropped)
                    det_result['ocr'] = ocr_results
                except Exception as e:
                    logger.error('OCR failed for detection {} in instance {}: {}', det, idx, e)
                    det_result['ocr'] = []
            prediction['detections'].append(det_result)
        predictions.append(prediction)
    # Log a summary
    total_boxes = sum(len(p['detections']) for p in predictions)
    logger.info(
        'Processed {} instances; total {} detections across all images', len(request.instances), total_boxes
    )
    return PredictionResponse(predictions=predictions)


@app.get('/')
def root() -> Dict[str, str]:
    '''Simple root endpoint that describes the service.'''
    return {
        'message': 'Vertex AI OCR service is running.',
        'health': AIP_HEALTH_ROUTE,
        'predict': AIP_PREDICT_ROUTE,
    }


if __name__ == '__main__':  # pragma: no cover
    logger.info('Starting local server on port {}', AIP_HTTP_PORT)
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=AIP_HTTP_PORT)

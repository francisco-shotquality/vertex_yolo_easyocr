"""
Unit tests for the YOLO + EasyOCR Vertex AI service.

These tests are written using Python's built‑in ``unittest`` framework so
they can be executed without installing external packages.  They
exercise the FastAPI application defined in ``src/main.py`` by making
HTTP requests against the in‑process app and by monkeypatching the
YOLO and EasyOCR model loaders.  Running these tests verifies that
the health check endpoint works and that the prediction endpoint
properly decodes inputs and returns the expected JSON structure.
"""

import base64
import unittest
from typing import List, Any

from fastapi.testclient import TestClient

# The application relies on the ``loguru`` logger, which is not installed in
# this environment.  Before importing the app module, insert a dummy
# implementation into ``sys.modules`` so that ``from loguru import logger``
# succeeds.  The dummy logger delegates to the built‑in ``logging`` module.
import logging
import sys
import types

if 'loguru' not in sys.modules:
    dummy_logger = logging.getLogger('dummy_loguru')
    dummy_loguru = types.SimpleNamespace(logger=dummy_logger)
    sys.modules['loguru'] = dummy_loguru

# Provide dummy versions of third‑party modules that may not be
# installed in this testing environment.  ``ultralytics`` provides
# the YOLO class used for object detection.  ``easyocr`` provides the
# Reader class for OCR.  Without these packages the application
# raises errors at import time, so we insert simple stubs into
# ``sys.modules`` before importing the app module.  These stubs are
# overwritten in individual tests where needed.
if 'ultralytics' not in sys.modules:
    class _DummyYOLO:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def predict(self, source: Any, conf: float = 0.25) -> list:
            return []

    sys.modules['ultralytics'] = types.SimpleNamespace(YOLO=_DummyYOLO)

if 'easyocr' not in sys.modules:
    class _DummyReader:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def readtext(self, image: Any) -> list:
            return []

    sys.modules['easyocr'] = types.SimpleNamespace(Reader=_DummyReader)

from vertex_yolo_easyocr.src import main as app_module


class DummyBoxes:
    """Simple container emulating the attributes of YOLO result boxes."""

    # Each of these attributes must be iterable.  ``xyxy`` holds
    # bounding boxes, ``conf`` holds confidence scores, and ``cls`` holds
    # class indices.
    xyxy: List[List[float]] = [[0.0, 0.0, 10.0, 10.0]]
    conf: List[float] = [0.99]
    cls: List[int] = [0]


class DummyResult:
    """Represents a single prediction result from a YOLO model."""
    boxes: Any = DummyBoxes()


class DummyYOLO:
    """Minimal YOLO model stub that returns fixed bounding boxes."""

    def predict(self, source: Any, conf: float = 0.25) -> List[DummyResult]:
        # Return one DummyResult per input image
        return [DummyResult() for _ in source]


class DummyOCR:
    """Minimal OCR stub that returns a fixed text annotation."""

    def readtext(self, image: Any) -> List[tuple]:
        # Each entry is (bbox, text, confidence)
        return [([0, 0, 10, 10], "dummy", 0.95)]


class VertexYoloEasyOCRTests(unittest.TestCase):
    """Test cases for the FastAPI app in ``src/main.py``."""

    def setUp(self) -> None:
        # Use TestClient to simulate HTTP requests to the FastAPI app
        self.client = TestClient(app_module.app)

    def test_health_endpoint(self) -> None:
        """
        Ensure that the health endpoint returns a 200 status and a payload
        indicating that the service is healthy.  The FastAPI app uses the
        string ``"healthy"`` to denote that the models are loaded and
        ready to serve requests.
        """
        response = self.client.get(app_module.AIP_HEALTH_ROUTE)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsInstance(body, dict)
        self.assertEqual(body.get("status"), "healthy")

    def test_predict_endpoint_with_stubs(self) -> None:
        """
        Verify that the prediction endpoint returns the expected JSON
        structure when using dummy YOLO and OCR models.  This test does
        not require the actual model weights and runs entirely in
        memory.
        """
        # Base64 encoded 1x1 pixel PNG image (white pixel)
        # Base64 encoded 1x1 white pixel PNG generated via Pillow
        image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
        # Save original functions and model ready flag
        original_models_ready = app_module._models_ready
        original_run_detection = app_module._run_detection
        original_run_ocr = app_module._run_ocr
        try:
            # Force models to be considered ready so that the endpoint responds
            app_module._models_ready = True

            # Patch detection to return a single box with known values
            def dummy_detection(image: Any, confidence: float = 0.5) -> List[dict]:
                return [
                    {
                        "xmin": 0.0,
                        "ymin": 0.0,
                        "xmax": 10.0,
                        "ymax": 10.0,
                        "confidence": 0.9,
                        "class_id": 0,
                        "class_name": "object",
                    }
                ]

            # Patch OCR to return a single text annotation
            def dummy_ocr(cropped: Any) -> List[dict]:
                return [
                    {
                        "text": "dummy",
                        "confidence": 0.95,
                        "bbox": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                    }
                ]

            app_module._run_detection = dummy_detection  # type: ignore[assignment]
            app_module._run_ocr = dummy_ocr  # type: ignore[assignment]

            # Send a POST request with a single instance
            payload = {"instances": [{"image": image_base64}]}
            response = self.client.post(app_module.AIP_PREDICT_ROUTE, json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            # Ensure top-level predictions list
            self.assertIn("predictions", data)
            predictions = data["predictions"]
            self.assertIsInstance(predictions, list)
            self.assertEqual(len(predictions), 1)
            pred = predictions[0]
            # Each prediction contains detections and a count
            self.assertIn("detections", pred)
            self.assertIn("detection_count", pred)
            detections = pred["detections"]
            self.assertEqual(pred["detection_count"], len(detections))
            # Check the structure of the first detection
            self.assertEqual(len(detections), 1)
            det = detections[0]
            self.assertIn("bbox", det)
            self.assertIn("class_name", det)
            self.assertIn("confidence", det)
            self.assertIn("ocr", det)
            self.assertEqual(det["class_name"], "object")
            self.assertEqual(det["ocr"][0]["text"], "dummy")
        finally:
            # Restore patched functions and state
            app_module._models_ready = original_models_ready
            app_module._run_detection = original_run_detection  # type: ignore[assignment]
            app_module._run_ocr = original_run_ocr  # type: ignore[assignment]


if __name__ == "__main__":
    # Allow the tests to be run directly via ``python test_main.py``
    unittest.main()

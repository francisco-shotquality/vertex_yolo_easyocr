# Vertex AI OCR Endpoint with YOLO and EasyOCR

This repository contains a complete example of how to build and deploy a
custom container to [Vertex AI](https://cloud.google.com/vertex-ai) that
performs two‑stage OCR on images.  The service first detects regions
of interest (ROIs) using a pre‑trained YOLO object detector and then
applies EasyOCR to recognize the text inside those regions.  The
inference server is implemented with [FastAPI](https://fastapi.tiangolo.com/) and
supports batch prediction out of the box.  A Dockerfile is provided
that uses an NVIDIA CUDA base image so that inference can run on
GPUs.  Bash scripts automate downloading model weights, building and
deploying the container to Vertex AI, and invoking the endpoint.

## Motivation

While Google’s managed services offer high‑level OCR APIs, some
applications require a custom pipeline: object detection to crop
interesting regions followed by text recognition.  Leveraging
Ultralytics YOLO for detection and EasyOCR for recognition combines
state‑of‑the‑art models while keeping the code simple.  Vertex AI’s
custom containers allow you to package this pipeline with complete
control over dependencies and runtime environment.  The `pyproject.toml`
file in this repository defines the necessary dependencies and uses
the `uv` package manager to install them efficiently【612638540015317†L236-L254】.

## Project structure

```
vertex_yolo_easyocr/
├── Dockerfile              # Custom container definition
├── pyproject.toml          # Dependency and build configuration【612638540015317†L236-L254】
├── README.md               # Documentation and usage instructions
├── models/                 # Placeholder for YOLO weights (downloaded separately)
├── scripts/
│   ├── build_and_deploy.sh # Build, push and deploy the container to Vertex AI
│   ├── download_weights.sh # Helper to download weights from a GCS bucket
│   └── invoke_endpoint.py  # Example script for calling the endpoint
└── src/
    └── main.py             # FastAPI application (YOLO + EasyOCR)
```

### Dependencies

All Python dependencies are declared in `pyproject.toml`.  The
[Ultralytics deployment guide](https://docs.ultralytics.com/guides/vertex-ai-deployment-with-docker/)
recommends listing YOLO, FastAPI, Uvicorn and Pillow in your
project’s dependency configuration【612638540015317†L236-L254】.  EasyOCR,
PyTorch and Google’s AI Platform SDK are added here to support text
recognition and endpoint invocation.

### Models

The repository does not ship with YOLO weights.  Use the helper
script in `scripts/download_weights.sh` to copy a weight file from a
Google Cloud Storage bucket into the `models/` directory:

```bash
bash scripts/download_weights.sh gs://your-bucket/path/to/yolov8n.pt models/yolov8n.pt
```

If you do not provide weights, the default value points at
`models/yolov8n.pt` which must exist at runtime.

## Local development

To test the service locally, first install the dependencies into a
virtual environment.  The `uv` command is a drop‑in replacement for
`pip` that performs parallel installation and builds wheels in
isolation.  Run the following commands from the root of the
repository:

```bash
# Create a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install uv and project dependencies
pip install uv
uv pip install -e .

# Download weights if necessary
bash scripts/download_weights.sh gs://your-bucket/path/to/yolov8n.pt models/yolov8n.pt

# Run the server locally
uvicorn src.main:app --host 0.0.0.0 --port 8080

# Test health check
curl http://localhost:8080/health

# Test prediction (example with a single image encoded as base64)
python scripts/invoke_endpoint.py \
  --project <PROJECT> --region <REGION> --endpoint-id dummy \
  --images path/to/your/image.jpg --confidence 0.3
```

In local mode the `--endpoint-id` argument to `invoke_endpoint.py`
is ignored; the script simply sends an HTTP POST to `localhost:8080`.  When
deployed to Vertex AI the same script calls the managed endpoint via
Google’s gRPC API.

## Building and deploying to Vertex AI

The repository includes a `Dockerfile` that bases the custom container
on `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`.  It installs
Python, the `uv` package manager and then uses `uv` to install the
project in editable mode.  The container exposes port 8080 and
configures the environment variables expected by Vertex AI.  Logging
is directed to stdout via Loguru to satisfy Vertex AI’s requirement
that logs be written to standard output【612638540015317†L396-L408】.

To automate building, pushing and deploying the container, run the
provided script:

```bash
bash scripts/build_and_deploy.sh \
  PROJECT_ID=<your-gcp-project> \
  REGION=us-central1 \
  REPOSITORY=vertex-yolo-easyocr \
  IMAGE_NAME=yolo-easyocr \
  TAG=$(git rev-parse --short HEAD) \
  MODEL_DISPLAY_NAME=yolo_easyocr_model \
  ENDPOINT_DISPLAY_NAME=yolo_easyocr_endpoint
```

This script performs the following actions:

1. **Ensure the Artifact Registry repository exists.**  It creates
   the specified Docker repository if necessary.
2. **Build the Docker image.**  The image is tagged with the fully
   qualified registry URI.
3. **Authenticate Docker and push the image.**  It calls
   `gcloud auth configure-docker` and pushes the image to Artifact
   Registry.
4. **Upload the model to Vertex AI.**  The model is registered with
   the container image URI and the health/predict routes.
5. **Create an endpoint if needed.**  If an endpoint with the given
   display name does not exist, it creates one.
6. **Deploy the model to the endpoint.**  The script attaches the
   latest model version to the endpoint, requesting a machine type and
   GPU accelerator (default: one T4 GPU).  You can modify
   `MACHINE_TYPE`, `ACCELERATOR_TYPE` and `ACCELERATOR_COUNT` in the
   script to match your requirements.

After deployment completes, the endpoint will be ready to receive
prediction requests.

## Invoking the endpoint

Use `scripts/invoke_endpoint.py` to send images to your deployed
endpoint.  The script accepts a list of image paths and optional
parameters such as a confidence threshold and a flag to skip OCR.  You
must authenticate with Google Cloud (e.g., via `gcloud auth
application-default login`) before calling the endpoint.

Example:

```bash
python scripts/invoke_endpoint.py \
  --project <PROJECT_ID> \
  --region us-central1 \
  --endpoint-name yolo_easyocr_endpoint \
  --images image1.jpg image2.jpg \
  --confidence 0.4
```

The script will print the JSON response returned by Vertex AI.  Each
prediction contains an array of detections with bounding boxes, class
names, detection confidence and the OCR results for each region.

## Notes and best practices

* **Batch prediction:**  The prediction endpoint accepts a list of
  instances, allowing you to process multiple images in a single
  request.  Vertex AI charges per prediction call and per node hour,
  so batching can reduce overhead.
* **Model weights:**  The choice of YOLO weights affects both the
  speed and accuracy of ROI detection.  Lightweight models like
  `yolov8n.pt` are fast but less accurate, whereas larger models
  (e.g., `yolov8x.pt`) may require more GPU memory.  The provided
  download script makes it easy to swap weights.
* **OCR languages:**  Set the `OCR_LANGS` environment variable to a
  comma‑separated list of language codes (e.g., `en,es,pt`) to have
  EasyOCR recognize multiple languages.  Only the necessary models
  will be loaded.
* **Error handling:**  The server returns HTTP 503 if the models have
  not finished loading.  During inference, exceptions are logged and
  a best‑effort response is returned.  Customize the error handling
  logic in `src/main.py` as needed.

## License and attribution

This project includes third‑party dependencies.  Ultralytics YOLO
models are licensed under AGPL‑3.0, which imposes sharing
requirements when you deploy modified versions of the model【612638540015317†L232-L234】.
Review the Ultralytics documentation and consult your legal team if
unsure.  EasyOCR is licensed under the Apache 2.0 license.  All code
in this repository is provided under the MIT License unless noted
otherwise.

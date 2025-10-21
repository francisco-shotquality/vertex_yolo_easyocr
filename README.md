# Vertex AI OCR Endpoint with YOLO and EasyOCR

This repository demonstrates how to build, deploy and invoke a custom container on Google Vertex AI for optical character recognition (OCR). The pipeline first detects regions of interest (ROIs) in input images using a pretrained YOLO object detector and then runs EasyOCR on each ROI to extract text. Running the object detector before OCR improves accuracy on cluttered images and allows the model to ignore irrelevant regions. The FastAPI application supports batch prediction and is designed to run on GPUs via an NVIDIA CUDA base image. Bash scripts automate downloading model weights, building and deploying the container, and calling the deployed Vertex AI endpoint.

## Motivation

Sometimes generic OCR services struggle with images that contain multiple objects or dense backgrounds. Detecting the regions that actually contain text before running OCR can significantly improve results. Using YOLO for detection and EasyOCR for text recognition allows you to assemble a custom two‑stage OCR pipeline tailored to your use case and provides full control over dependencies and runtime environment.

## Dependencies

All runtime dependencies are declared in `pyproject.toml` and installed automatically by the [uv](https://github.com/astral-sh/uv) package manager. The core libraries include:

- [ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 object detection
- [easyocr](https://github.com/JaidedAI/EasyOCR) for text recognition
- [fastapi](https://github.com/tiangolo/fastapi) and [uvicorn](https://github.com/encode/uvicorn) for the web server
- [loguru](https://github.com/Delgan/loguru) for structured logging
- `python-dotenv` for loading environment variables

## Local development and testing

You can build and run the inference service locally using Docker. First make sure you have downloaded the YOLO model weights (see `scripts/download_weights.sh`), then build the container image from the repository root:

```bash
docker build -t vertex-yolo-easyocr .
```

Run the container and expose the HTTP port:

```bash
docker run --env-file .env -p 8080:8080 vertex-yolo-easyocr
```

The service will start a FastAPI app listening on port 8080. You can send prediction requests to `http://localhost:8080/predict` using the example script in `scripts/invoke_endpoint.py` or any HTTP client. Environment variables such as the YOLO weights path and OCR languages can be set in your `.env` file and are loaded automatically by the application.

## Building and deploying to Vertex AI

To deploy the service on Vertex AI custom containers, you must first have a Google Cloud project with Vertex AI and Artifact Registry enabled. Populate the required deployment variables (project ID, region, repository, image name, etc.) in your `.env` file. Then run the build and deploy script:

```bash
bash scripts/build_and_deploy.sh
```

This script:

1. Builds the Docker image and pushes it to your Artifact Registry repository
2. Creates or updates a model in Vertex AI pointing to the container image
3. Deploys the model to an endpoint in Vertex AI with GPU support

The script uses environment variables defined in `.env` to parameterize the deployment. See the comments inside `scripts/build_and_deploy.sh` for details.

## Invoking the endpoint

After deployment, you can invoke the Vertex AI endpoint programmatically. The example script `scripts/invoke_endpoint.py` takes the endpoint ID, image file, and project information and sends a prediction request:

```bash
python scripts/invoke_endpoint.py \
  --project_id your-project-id \
  --endpoint_id your-endpoint-id \
  --region your-region \
  --image_path path/to/your/image.jpg
```

The script encodes the image, calls the endpoint via the Vertex AI Python SDK and prints the detection results.

## Notes and best practices

- Use GPU-enabled machine types (e.g., `n1-standard-8` with an attached NVIDIA T4 or A100) when deploying the endpoint for best performance.
- Place your YOLO weights file in the `models` directory or configure an alternate path in your `.env` file.
- For large batches of images, adjust the request payload size and concurrency settings in the `predict` endpoint accordingly.

## License and attribution

The YOLOv8 model from Ultralytics is licensed under the GNU Affero General Public License v3.0 (AGPL‑3.0). Please ensure that your use complies with the terms of this license. EasyOCR is licensed under the Apache License 2.0. See the respective repositories for full licensing details.

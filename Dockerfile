FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-distutils \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# Ensure python points to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install uv package manager to handle dependencies
RUN pip install --no-cache-dir uv

# Set the working directory to /app inside the container
WORKDIR /app

# Copy project files into the container.  We do this in two stages to
# leverage Docker layer caching: first copy only the dependency file,
# install dependencies, and then copy the rest of the code.  This
# ensures that changes to your code do not invalidate the heavy
# dependency installation layer.
COPY pyproject.toml ./

# Install Python dependencies using uv.  The `--system-site-packages` flag
# is not used here to ensure a clean environment.  The `-e .` flag
# installs the current package in editable mode so that the source
# remains importable from /app/src.  uv caches wheels internally
# improving build performance.
RUN uv pip install --no-cache-dir -e .

# Copy the rest of the repository (source code, scripts, models, etc.)
COPY . .

# Expose the default port used by Vertex AI.  The actual port is
# controlled by the AIP_HTTP_PORT environment variable at runtime.
EXPOSE 8080

# Set environment variables used by the inference server.  These can
# be overridden when deploying the container on Vertex AI.  The path
# to the YOLO weights defaults to a file in the `models` folder.
ENV AIP_HTTP_PORT=8080 \
    AIP_HEALTH_ROUTE=/health \
    AIP_PREDICT_ROUTE=/predict \
    YOLO_WEIGHTS_PATH=/app/models/yolov8n.pt \
    OCR_LANGS=en

# Command to start the FastAPI server using Uvicorn.  Vertex AI
# overrides the port at runtime via the AIP_HTTP_PORT environment
# variable.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]

#!/usr/bin/env bash
# Build the Docker image, push it to Artifact Registry and deploy to Vertex AI.
#
# This script streamlines the end‑to‑end deployment of the custom
# container defined in this repository.  It assumes that the Google
# Cloud CLI (`gcloud`) is installed and authenticated, and that you
# have permission to create Artifact Registry repositories, upload
# models and create Vertex AI endpoints in your project.
#
# Environment variables control most aspects of the deployment.  The
# defaults are reasonable for a first test but should be customized for
# production use.

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Google Cloud project ID.  Falls back to the active gcloud project.
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"

# The GCP region to deploy the model and endpoint.  Must support the
# accelerators you intend to use.  Common choices include us‑central1,
# us‑east1, europe‑west4, etc.
REGION="${REGION:-us-central1}"

# Artifact Registry repository name.  This repository must exist or
# will be created automatically by gcloud below.
REPOSITORY="${REPOSITORY:-vertex-yolo-easyocr}"

# Docker image name and tag.  Use semantic versioning or a commit
# hash for reproducible deployments.
IMAGE_NAME="${IMAGE_NAME:-yolo-easyocr}"
IMAGE_TAG="${TAG:-latest}"

# Display name for the uploaded model and endpoint.  This name is
# visible in the Vertex AI console and may be reused across versions.
MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME:-yolo_easyocr_model}"
ENDPOINT_DISPLAY_NAME="${ENDPOINT_DISPLAY_NAME:-yolo_easyocr_endpoint}"

# Machine type and accelerator configuration for the endpoint.  Adjust
# according to your inference workload and budget.  The example below
# requests a single T4 GPU.  Consult Vertex AI documentation for
# supported types and regions.
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_TESLA_T4}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"

# Path to the local directory containing the Dockerfile.  Default is
# the root of this repository (the directory containing this script).
CONTEXT_DIR="${CONTEXT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# Artifact Registry domain for the selected region.  See:
# https://cloud.google.com/artifact-registry/docs/docker/store-docker-images#configure
REGISTRY_DOMAIN="${REGION}-docker.pkg.dev"

# Fully qualified image URI (e.g. us-central1-docker.pkg.dev/my-project/my-repo/my-image:tag)
IMAGE_URI="${REGISTRY_DOMAIN}/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

function ensure_repository() {
    # Create the Artifact Registry repository if it does not exist.
    if ! gcloud artifacts repositories describe "${REPOSITORY}" \
        --location="${REGION}" >/dev/null 2>&1; then
        echo "Creating Artifact Registry repository ${REPOSITORY} in ${REGION}"
        gcloud artifacts repositories create "${REPOSITORY}" \
            --repository-format=docker \
            --location="${REGION}" \
            --description="Repository for YOLO EasyOCR custom containers"
    fi
}

function build_image() {
    echo "Building Docker image ${IMAGE_URI} from ${CONTEXT_DIR}"
    docker build -t "${IMAGE_URI}" "${CONTEXT_DIR}"
}

function push_image() {
    echo "Authenticating Docker with Artifact Registry"
    gcloud auth configure-docker "${REGISTRY_DOMAIN}" --quiet
    echo "Pushing image ${IMAGE_URI}"
    docker push "${IMAGE_URI}"
}

function upload_model() {
    echo "Uploading model to Vertex AI using image ${IMAGE_URI}"
    gcloud ai models upload "${MODEL_DISPLAY_NAME}" \
        --region="${REGION}" \
        --container-image-uri="${IMAGE_URI}" \
        --container-ports="${AIP_HTTP_PORT:-8080}" \
        --container-health-route="${AIP_HEALTH_ROUTE:-/health}" \
        --container-predict-route="${AIP_PREDICT_ROUTE:-/predict}"
}

function create_endpoint_if_needed() {
    # Create the endpoint if one with the display name does not exist.
    local existing
    existing=$(gcloud ai endpoints list --region="${REGION}" \
        --filter="display_name=${ENDPOINT_DISPLAY_NAME}" \
        --format="value(name)" | head -n1 || true)
    if [[ -z "${existing}" ]]; then
        echo "Creating endpoint ${ENDPOINT_DISPLAY_NAME}"
        gcloud ai endpoints create \
            --region="${REGION}" \
            --display-name="${ENDPOINT_DISPLAY_NAME}"
    else
        echo "Endpoint already exists: ${existing}"
    fi
}

function deploy_model() {
    # Retrieve the model and endpoint IDs
    local model_id
    model_id=$(gcloud ai models list --region="${REGION}" \
        --filter="display_name=${MODEL_DISPLAY_NAME}" \
        --sort-by="create_time" \
        --format="value(name)" | tail -n1)
    local endpoint_id
    endpoint_id=$(gcloud ai endpoints list --region="${REGION}" \
        --filter="display_name=${ENDPOINT_DISPLAY_NAME}" \
        --format="value(name)" | head -n1)
    if [[ -z "${model_id}" || -z "${endpoint_id}" ]]; then
        echo "ERROR: Could not determine model or endpoint IDs." >&2
        exit 1
    fi
    echo "Deploying model ${model_id} to endpoint ${endpoint_id}"
    gcloud ai endpoints deploy-model "${endpoint_id}" \
        --region="${REGION}" \
        --model="${model_id}" \
        --display-name="${MODEL_DISPLAY_NAME}-deployment" \
        --machine-type="${MACHINE_TYPE}" \
        --accelerator-type="${ACCELERATOR_TYPE}" \
        --accelerator-count="${ACCELERATOR_COUNT}" \
        --min-replica-count=1 \
        --max-replica-count=1 \
        --traffic-split=0=100
}

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

echo "Using project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Repository: ${REPOSITORY}"
echo "Image URI: ${IMAGE_URI}"
echo "Model display name: ${MODEL_DISPLAY_NAME}"
echo "Endpoint display name: ${ENDPOINT_DISPLAY_NAME}"
ensure_repository
build_image
push_image
upload_model
create_endpoint_if_needed
deploy_model

echo "Deployment complete.  Your endpoint is ready to receive requests."

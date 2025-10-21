#!/usr/bin/env bash
# Download a YOLO model weight file from a Google Cloud Storage bucket.
#
# This script is a convenience helper for retrieving model weights used by
# the inference server.  It wraps `gsutil cp` to copy a file from a
# Google Cloud Storage (GCS) URI to a local destination.  The
# destination directory is created automatically if it does not exist.
#
# Usage:
#   bash download_weights.sh gs://<bucket>/<path>/<file>.pt [destination]
#
# Example:
#   bash download_weights.sh gs://my-bucket/models/yolov8n.pt models/yolov8n.pt
#
# If the destination argument is omitted, the weight file will be
# downloaded into the `models/` directory with its original filename.

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 GCS_PATH [DESTINATION]" >&2
    exit 1
fi

GCS_PATH="$1"
DEST="${2:-}"  # Optional destination

if [[ -z "$DEST" ]]; then
    FILENAME="$(basename "$GCS_PATH")"
    DEST="models/$FILENAME"
fi

# Ensure the destination directory exists
mkdir -p "$(dirname "$DEST")"

echo "Downloading $GCS_PATH to $DEST"
# Use gsutil to copy the file.  The gsutil CLI must be installed and
# authenticated.  Consult Google Cloud documentation for details.
gsutil cp "$GCS_PATH" "$DEST"
echo "Download completed."

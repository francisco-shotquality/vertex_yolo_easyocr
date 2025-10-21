#!/usr/bin/env python
'''Example script for invoking the deployed Vertex AI OCR endpoint.

This script demonstrates how to call a Vertex AI endpoint created
from this project. It accepts one or more image file paths, encodes
them as base64 strings, and sends them as a batch to the endpoint.
The response is printed to stdout.

Usage:
    python invoke_endpoint.py --project <PROJECT_ID> --region <REGION> --endpoint-id <ENDPOINT_ID> --images path1.jpg path2.jpg

Alternatively, you can specify the endpoint display name instead of the numeric ID. When using a display name, the script will look up the endpoint ID via the Vertex AI API. Provide either --endpoint-id or --endpoint-name.

Requires the google-cloud-aiplatform library, which is included in the project's dependencies. You must also have application default credentials (ADC) configured, such as via gcloud auth application-default login.
'''

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import List

from google.cloud import aiplatform


def encode_image(path: Path) -> str:
    '''Read a binary image file and return a base64 string.'''
    with path.open('rb') as f:
        data = f.read()
    return base64.b64encode(data).decode('utf-8')


def get_endpoint_id(display_name: str, project: str, region: str) -> str:
    '''Resolve a Vertex AI endpoint ID from its display name.'''
    client = aiplatform.gapic.EndpointServiceClient(client_options={'api_endpoint': f'{region}-aiplatform.googleapis.com'})
    parent = f'projects/{project}/locations/{region}'
    filter_str = 'display_name=' + display_name
    response = client.list_endpoints(parent=parent, filter=filter_str)
    for endpoint in response:
        return endpoint.name.split('/')[-1]
    raise ValueError(f'No endpoint found with display name {display_name}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Invoke Vertex AI OCR endpoint')
    parser.add_argument('--project', required=True, help='GCP project ID')
    parser.add_argument('--region', required=True, help='Region where the endpoint is deployed')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--endpoint-id', help='Numeric endpoint ID')
    group.add_argument('--endpoint-name', help='Endpoint display name')
    parser.add_argument(
        '--images',
        nargs='+',
        required=True,
        help='One or more image file paths to send to the endpoint',
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=None,
        help='Optional confidence threshold to override the default on the server',
    )
    parser.add_argument(
        '--skip-ocr',
        action='store_true',
        help='If set, the server will skip running OCR on the detected regions',
    )
    args = parser.parse_args()

    if args.endpoint_id:
        endpoint_id = args.endpoint_id
    else:
        endpoint_id = get_endpoint_id(args.endpoint_name, args.project, args.region)
        print(f'Resolved endpoint {args.endpoint_name} to ID {endpoint_id}')

    # Initialize the Vertex AI Endpoint resource
    aiplatform.init(project=args.project, location=args.region)
    endpoint = aiplatform.Endpoint(endpoint_id=endpoint_id)

    # Build instances
    instances: List[dict] = []
    for img_path in args.images:
        path = Path(img_path)
        if not path.exists():
            raise FileNotFoundError(f'Image not found: {img_path}')
        encoded = encode_image(path)
        instances.append({'image': encoded})

    parameters = {}
    if args.confidence is not None:
        parameters['confidence'] = args.confidence
    if args.skip_ocr:
        parameters['run_ocr'] = False

    print('Sending request to endpoint...')
    response = endpoint.predict(instances=instances, parameters=parameters)
    print(json.dumps(response, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Test script for the GA stock prediction deployed endpoint.

Usage:
    python tests/test_endpoint.py

Requirements:
    pip install azure-ai-ml azure-identity
"""

import json
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# ── Configure these to match your Azure workspace ──────────────────────────────
SUBSCRIPTION_ID = "<your-subscription-id>"
RESOURCE_GROUP  = "<your-resource-group>"
WORKSPACE_NAME  = "<your-workspace-name>"
ENDPOINT_NAME   = "ga-deployment"
DEPLOYMENT_NAME = "green"          # optional: target a specific deployment
# ───────────────────────────────────────────────────────────────────────────────

# The scoring script's run() function expects:
#   A JSON string whose values are Close prices (floats).
#   Minimum 5 values required (LSTM lookback_size = 5).
SAMPLE_PAYLOAD = json.dumps({
    "0": 150.25,
    "1": 152.10,
    "2": 149.80,
    "3": 153.45,
    "4": 151.90,
    "5": 154.20,
    "6": 156.75,
    "7": 155.30,
    "8": 157.80,
    "9": 159.45,
})

def main():
    credential = DefaultAzureCredential()
    client = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

    print(f"Invoking endpoint '{ENDPOINT_NAME}' (deployment: '{DEPLOYMENT_NAME}')...")
    response = client.online_endpoints.invoke(
        endpoint_name=ENDPOINT_NAME,
        request_file=None,          # we pass the body directly below
        deployment_name=DEPLOYMENT_NAME,
        # Pass raw JSON string — matches what run(raw_data) expects
        request_file=None,
    )

    # Alternative: use the request_file approach with sample_payload.json
    # response = client.online_endpoints.invoke(
    #     endpoint_name=ENDPOINT_NAME,
    #     request_file="tests/sample_payload.json",
    #     deployment_name=DEPLOYMENT_NAME,
    # )

    print("Prediction result:", response)

if __name__ == "__main__":
    main()

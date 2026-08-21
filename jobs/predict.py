"""Client inference script to test Azure ML Managed Online Endpoint or local scoring."""

import argparse
import json
import subprocess
import requests


def predict_endpoint(endpoint_name: str, sample_prices: list[float]) -> None:
    """Send stock price sequence to the Azure ML Online Endpoint."""
    payload = {"data": sample_prices}
    payload_json = json.dumps(payload)
    
    print(f"Sending input prices to endpoint '{endpoint_name}': {sample_prices}")

    # Method 1: Via Azure ML CLI invoke (automatic auth)
    cmd = [
        "az", "ml", "online-endpoint", "invoke",
        "--name", endpoint_name,
        "--request-file", "-"
    ]
    try:
        res = subprocess.run(cmd, input=payload_json, capture_output=True, text=True, check=True)
        print("Prediction Result (Azure ML CLI Invoke):", res.stdout)
    except Exception as err:
        print(f"CLI invoke failed or az CLI not authenticated locally: {err}")
        print("Tip: You can also call the endpoint directly over HTTP REST API with your Primary Key.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test inference against Azure ML Online Endpoint.")
    parser.add_argument("--endpoint", default="ga-deployment", help="Name of the online endpoint.")
    parser.add_argument(
        "--prices",
        nargs="+",
        type=float,
        default=[410.50, 412.30, 415.00, 413.80, 418.20],
        help="Space-separated list of last 5 days closing prices.",
    )

    args = parser.parse_args()
    predict_endpoint(args.endpoint, args.prices)

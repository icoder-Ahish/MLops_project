#!/usr/bin/env python3
"""Monitor an Azure ML stock endpoint for performance and input drift.

The monitor evaluates predictions only where the next market close is already
known. That avoids treating an unavailable future label as a model failure.
It reports two independent signals:
* MAPE: recent prediction performance (model degradation).
* PSI: shift in the recent Close-price distribution versus training data.

Exit codes: 0 healthy, 2 drift/degradation detected and alerted, 1 execution or
alert-delivery failure.
"""

import argparse
import json
import os
import smtplib
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import yaml


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Return PSI, with stable bins and smoothing for empty buckets."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("PSI requires non-empty reference and current samples.")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(reference, current) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    expected, _ = np.histogram(reference, bins=edges)
    actual, _ = np.histogram(current, bins=edges)
    epsilon = 1e-6
    expected_pct = np.maximum(expected / expected.sum(), epsilon)
    actual_pct = np.maximum(actual / actual.sum(), epsilon)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def invoke_endpoint(endpoint_name: str, prices: list[float]) -> float:
    """Invoke the endpoint through the authenticated Azure CLI and return its last prediction."""
    payload = {str(index): float(value) for index, value in enumerate(prices)}
    request_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as request_file:
            json.dump(payload, request_file)
            request_path = request_file.name
        completed = subprocess.run(
            [
                "az", "ml", "online-endpoint", "invoke", "--name", endpoint_name,
                "--request-file", request_path, "--output", "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        response = json.loads(completed.stdout)
        values = np.asarray(response, dtype=float).reshape(-1)
        if values.size == 0:
            raise ValueError("Endpoint returned no predictions.")
        return float(values[-1])
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.strip() or "Azure ML endpoint invocation failed.") from error
    finally:
        if request_path:
            Path(request_path).unlink(missing_ok=True)


def download_closes(ticker: str, minimum_rows: int) -> pd.Series:
    start = date.today() - timedelta(days=max(90, minimum_rows * 3))
    frame = yf.download(ticker, start=start.isoformat(), progress=False, auto_adjust=True)
    if frame.empty or "Close" not in frame:
        raise RuntimeError(f"No Close-price data returned for {ticker}.")
    closes = frame["Close"]
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    return closes.dropna().astype(float)


def recent_mape(
    endpoint_name: str,
    closes: pd.Series,
    lookback_days: int,
    windows: int,
    prediction_horizon: int,
) -> float:
    """Evaluate each forecast against its known target observation."""
    required = lookback_days + windows + prediction_horizon
    if len(closes) < required:
        raise ValueError(f"Need at least {required} closes; received {len(closes)}.")
    values = closes.to_numpy(dtype=float)
    predictions, actuals = [], []
    first_window = len(values) - windows - prediction_horizon
    for start in range(first_window, len(values) - prediction_horizon):
        history = values[start - lookback_days + 1 : start + 1]
        predictions.append(invoke_endpoint(endpoint_name, history.tolist()))
        actuals.append(values[start + prediction_horizon])
    actuals_array = np.asarray(actuals)
    return float(np.mean(np.abs((actuals_array - np.asarray(predictions)) / actuals_array)))


def send_email(result: dict, smtp_host: str, smtp_port: int, username: str, password: str, sender: str, recipients: str) -> None:
    if not all([smtp_host, username, password, sender, recipients]):
        raise RuntimeError("Drift detected, but SMTP alert secrets are not fully configured.")
    message = EmailMessage()
    message["Subject"] = f"[ACTION REQUIRED] Model drift: {result['endpoint_name']}"
    message["From"] = sender
    message["To"] = recipients
    message.set_content(
        "Model monitoring detected a threshold breach.\n\n"
        f"Endpoint: {result['endpoint_name']}\nTicker: {result['ticker']}\n"
        f"MAPE: {result['mape']:.2%} (limit: {result['max_mape']:.2%})\n"
        f"PSI: {result['psi']:.4f} (limit: {result['max_psi']:.4f})\n"
        f"Breaches: {', '.join(result['breaches'])}\n\n"
        "Investigate the endpoint and recent market data, then retrain/redeploy if appropriate."
    )
    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/model_monitoring.yml")
    parser.add_argument("--output", default="monitoring-result.json")
    parser.add_argument("--smtp-host", default=os.getenv("SMTP_HOST", ""))
    parser.add_argument("--smtp-port", type=int, default=int(os.getenv("SMTP_PORT", "587")))
    parser.add_argument("--smtp-username", default=os.getenv("SMTP_USERNAME", ""))
    parser.add_argument("--smtp-password", default=os.getenv("SMTP_PASSWORD", ""))
    parser.add_argument("--alert-from", default=os.getenv("ALERT_EMAIL_FROM", ""))
    parser.add_argument("--alert-to", default=os.getenv("ALERT_EMAIL_TO", ""))
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    reference = pd.read_csv(config["reference_data"])["Close"].dropna().to_numpy(dtype=float)
    windows = int(config["evaluation_windows"])
    lookback_days = int(config["lookback_days"])
    prediction_horizon = int(config["prediction_horizon"])
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon must be at least 1.")
    closes = download_closes(config["ticker"], lookback_days + windows + prediction_horizon)
    mape = recent_mape(
        config["endpoint_name"], closes, lookback_days, windows, prediction_horizon
    )
    psi = population_stability_index(reference, closes.tail(windows + lookback_days).to_numpy())
    breaches = []
    if mape > float(config["max_mape"]):
        breaches.append("MAPE exceeded its limit")
    if psi > float(config["max_psi"]):
        breaches.append("PSI exceeded its limit")
    result = {
        "endpoint_name": config["endpoint_name"], "ticker": config["ticker"],
        "mape": mape, "max_mape": float(config["max_mape"]),
        "prediction_horizon": prediction_horizon,
        "psi": psi, "max_psi": float(config["max_psi"]), "breaches": breaches,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not breaches:
        return 0
    send_email(result, args.smtp_host, args.smtp_port, args.smtp_username, args.smtp_password, args.alert_from, args.alert_to)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Monitoring failed: {error}", file=sys.stderr)
        sys.exit(1)

"""Download recent Yahoo Finance data and prepare its Azure ML data asset YAML."""

from datetime import date, datetime, timedelta
import argparse
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_UPLOAD_YAML = PROJECT_ROOT / "jobs" / "data_upload.yml"


def get_ticker_data(ticker: str, start_days: int, end_days: int) -> None:
    """Download daily data from ``start_days`` ago through ``end_days`` ago.

    Yahoo Finance treats ``end`` as exclusive. Adding one day makes an
    ``end_days`` value of zero include today's available trading data.
    """
    if start_days < end_days:
        raise ValueError("start must be greater than or equal to end.")

    today = date.today()
    start_date = today - timedelta(days=start_days)
    end_date = today - timedelta(days=end_days) + timedelta(days=1)
    ticker_data = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if ticker_data.empty:
        raise ValueError(f"No data found for {ticker}. Check the ticker symbol or retry later.")

    close = ticker_data["Close"].squeeze("columns").rename("Close")
    close.index.name = "Date"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / f"{ticker}.csv"
    close.to_csv(data_path)

    tags = get_dataset_tags(close)
    save_to_data_upload(data_path, ticker, tags)
    logging.info("Saved %s rows to %s", len(close), data_path)
    
def get_dataset_tags(data: pd.Series) -> dict[str, str | int | float]:
    """Return serializable data-quality tags for the Azure ML asset."""
    return {
        "Length": len(data),
        "Start": str(data.index[0].date()),
        "End": str(data.index[-1].date()),
        "Median": round(float(np.median(data)), 2),
        "SD": round(float(np.std(data, ddof=1)), 2),
    }

def save_to_data_upload(data_path: Path, ticker: str, tags: dict[str, str | int | float]) -> None:
    """Write a valid Azure ML data asset definition relative to the repository."""
    name = ticker.split(".", maxsplit=1)[0]
    version = datetime.now().strftime("%Y%m%d%H%M%S")
    # `jobs/data_upload.yml` lives in `jobs/`, so the asset path needs to go
    # up one level to reach the repository-root `data/` directory.
    relative_path = Path("..") / data_path.relative_to(PROJECT_ROOT)
    description = f"Stock data for {ticker} during {tags['Start']}:{tags['End']} in 1d interval."
    DATA_UPLOAD_YAML.write_text(
        f"$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json\n"
        f"type: uri_file\n"
        f"name: '{name}'\n"
        f"description: '{description}'\n"
        f"path: '{relative_path.as_posix()}'\n"
        "tags:\n"
        f"  Length: {tags['Length']}\n"
        f"  Start: '{tags['Start']}'\n"
        f"  End: '{tags['End']}'\n"
        f"  Median: {tags['Median']}\n"
        f"  SD: {tags['SD']}\n"
        f"version: '{version}'\n",
        encoding="utf-8",
    )

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ticker", required=True, help="Yahoo Finance ticker, e.g. WIPRO.NS.")
    parser.add_argument("--start", type=int, default=365, help="Calendar days before today to begin.")
    parser.add_argument("--end", type=int, default=0, help="Calendar days before today to end, inclusively.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    get_ticker_data(args.ticker, args.start, args.end)

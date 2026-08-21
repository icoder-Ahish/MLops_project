import numpy as np
import pandas as pd

from jobs import monitor_model


def test_psi_is_near_zero_for_matching_distributions():
    reference = np.arange(1, 101, dtype=float)
    assert monitor_model.population_stability_index(reference, reference.copy()) < 1e-8


def test_psi_detects_a_shifted_distribution():
    reference = np.arange(1, 101, dtype=float)
    shifted = np.arange(201, 301, dtype=float)
    assert monitor_model.population_stability_index(reference, shifted) > 0.2


def test_mape_uses_the_configured_prediction_horizon(monkeypatch):
    calls = []

    def fake_invoke(_endpoint, prices):
        calls.append(prices)
        return prices[-1] + 2

    monkeypatch.setattr(monitor_model, "invoke_endpoint", fake_invoke)
    closes = pd.Series(np.arange(1, 15, dtype=float))

    mape = monitor_model.recent_mape(
        "endpoint", closes, lookback_days=5, windows=2, prediction_horizon=2
    )

    assert len(calls) == 2
    assert mape == 0.0

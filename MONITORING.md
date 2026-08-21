# Model monitoring and alerting

This project monitors the deployed `ga-deployment` Azure ML endpoint after the
market closes on weekdays. The implementation is in
[`jobs/monitor_model.py`](jobs/monitor_model.py), and the schedule is in
[`.github/workflows/model_monitoring.yml`](.github/workflows/model_monitoring.yml).

## What is monitored

| Signal | What it answers | Alert condition |
| --- | --- | --- |
| Prediction MAPE | Has the deployed model's recent forecast accuracy degraded? | Recent MAPE is greater than `max_mape`. |
| Population Stability Index (PSI) | Have the endpoint input prices moved away from the training-price distribution? | PSI is greater than `max_psi`. |

The default configuration is in
[`config/model_monitoring.yml`](config/model_monitoring.yml): 20 evaluation
windows, a 5-day model lookback, a 2-observation forecast horizon, maximum
MAPE of 10%, and maximum PSI of 0.20.
These are initial operational limits, not universal values. Review the first
few healthy runs and tune them to the model's normal error and the desired
alert sensitivity.

## How model degradation is detected

The endpoint receives the preceding five closing prices. The current training
code uses a target two trading observations after the final input price, so the
configured `prediction_horizon` is `2`. The monitor downloads recent adjusted
closing prices for the configured ticker from Yahoo Finance. For each of the
most recent 20 dates for which the target close is already known, it:

1. Sends the five closes ending on day `t` to the deployed Azure ML endpoint.
2. Treats the actual close at `t + prediction_horizon` as the ground truth.
3. Calculates the absolute percentage error for that forecast:

   ```text
   APE = abs(actual - prediction) / abs(actual)
   ```

4. Averages the 20 errors:

   ```text
   MAPE = mean(APE for each evaluated window)
   ```

An alert is raised when `MAPE > max_mape`. With the default setting, a MAPE of
`0.12` is displayed as 12% and breaches the 10% limit.

Using only delayed, known closes is important: the monitor never labels a
prediction as wrong merely because its target market price does not yet exist.
This is performance monitoring of the deployed endpoint, not a
training-validation metric. If the training code is changed to forecast the
next close, set `prediction_horizon: 1` and retrain/redeploy before monitoring
that model.

## How data drift is detected

Data drift concerns the inputs, independently of whether recent predictions
happen to be accurate. The monitor compares:

- **Reference data:** the `Close` column in `data/WIPRO.NS.csv`, which is the
  configured training reference data.
- **Current data:** the latest 25 closes (20 evaluation windows plus the
  5-day lookback) downloaded for the configured ticker.

It creates ten quantile-based buckets from the reference values. For every
bucket it calculates the reference proportion `E` and current proportion `A`.
The Population Stability Index is then:

```text
PSI = sum((A - E) * ln(A / E)) for every bucket
```

The implementation adds a very small value (`1e-6`) to empty bucket shares so
the calculation remains finite. `PSI = 0` means the bucket distributions are
identical; a higher value means the recent input-price distribution differs
more from the training reference. An alert is raised when `PSI > max_psi`.
The default `0.20` is a commonly used investigation threshold; it should be
calibrated to this model and data rather than treated as a guarantee of poor
model performance.

## Alert behavior

The scheduled workflow runs at 13:00 UTC, Monday through Friday. On a breach
it writes `monitoring-result.json`, sends an SMTP email containing the endpoint,
ticker, MAPE, PSI, limits, and breached conditions, then exits with code `2`.
GitHub Actions displays that run as failed so the issue is visible even if mail
delivery is overlooked. Healthy runs exit with code `0` and do not send email.

Required GitHub secrets are:

- `SMTP_HOST`
- `SMTP_PORT` (normally `587`)
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `ALERT_EMAIL_FROM`
- `ALERT_EMAIL_TO`

If monitoring finds a breach but the SMTP settings are absent or invalid, the
workflow fails. It does not silently drop the alert.

## Responding to an alert

1. Open the workflow summary and inspect `monitoring-result.json`.
2. Confirm the endpoint is serving the expected model version and that Yahoo
   Finance returned valid market data.
3. If only PSI breached, inspect whether a market-level price regime change is
   expected. A PSI breach is an investigation signal, not proof of failure.
4. If MAPE breached, compare several endpoint predictions with actual closes
   and inspect the latest training data.
5. Retrain and deploy a new model when the degradation is confirmed, then
   review the thresholds after stable runs.

## Current scope and limitation

This repository's endpoint does not persist each production request, response,
or user-provided outcome. Therefore this monitor evaluates the live model on
recent, delayed market data rather than joining production inference logs to
labels. For a general-purpose production endpoint, add durable request,
prediction, model-version, timestamp, and eventual-label logging; calculate
these same metrics from that joined data instead.

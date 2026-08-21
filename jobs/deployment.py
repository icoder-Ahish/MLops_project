import pandas as pd
import numpy as np
import pickle
import torch
import os
import json
import logging
import pytorch_lightning as pl

# ─── Module-level class definitions ─────────────────────────────────────────
# IMPORTANT: These classes must be at module scope (not inside init()) so that
# torch.load() / pickle can deserialize the saved model state dict correctly.

class dataset(pl.LightningDataModule):

    def __init__(self, scaler):
        super(dataset, self).__init__()
        self.lookback_size = 5
        self.scaler = scaler

    def predict_tensors(self, df):
        X = []
        for i in np.arange(self.lookback_size, len(df) + 1):
            X.append(df[i - self.lookback_size:i])
        X = np.array(X).reshape(-1, self.lookback_size, 1)
        return torch.from_numpy(X).float()

    def predict_dataloader(self, data):
        self.pred_df = self.scaler.transform(data)
        self.pred_data = self.predict_tensors(self.pred_df)
        return self.pred_data


class model(pl.LightningModule):

    def __init__(self, lookback_size=5):
        super(model, self).__init__()
        self.lookback_size = lookback_size
        self.lstm = torch.nn.LSTM(batch_first=True, input_size=1, hidden_size=self.lookback_size)
        self.out = torch.nn.Linear(5, 1)
        self.loss = torch.nn.functional.mse_loss

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x)
        x = x[:, -1]
        x = self.out(x)
        return x, hidden

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, _ = batch
        return self(X.type(torch.float32))


# ─── Global inference objects ────────────────────────────────────────────────
mod = None
datamod = None
scaler = None


def init():
    """Called once when the container starts. Loads model and scaler into memory."""
    global mod, datamod, scaler

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
        logging.info("AZUREML_MODEL_DIR = %s", model_dir)
        logging.info("Files in model_dir: %s", os.listdir(model_dir))

        # ── Scaler ────────────────────────────────────────────────────────────
        scalerpath = os.path.join(model_dir, "outputs", "scaler.pkl")
        if not os.path.exists(scalerpath):
            scalerpath = os.path.join(model_dir, "scaler.pkl")
        logging.info("Loading scaler from: %s", scalerpath)
        with open(scalerpath, "rb") as f:
            scaler = pickle.load(f)

        # ── Model ─────────────────────────────────────────────────────────────
        modelpath = os.path.join(model_dir, "outputs", "model.pth")
        if not os.path.exists(modelpath):
            modelpath = os.path.join(model_dir, "model.pth")
        logging.info("Loading model from: %s", modelpath)
        mod = model()
        mod.load_state_dict(torch.load(modelpath, map_location=torch.device("cpu")))
        mod.eval()

        datamod = dataset(scaler)
        logging.info("init() completed successfully.")

    except Exception as e:
        logging.exception("init() failed with error: %s", str(e))
        raise  # Re-raise so Azure ML marks deployment as failed immediately


def run(raw_data):
    try:
        data_json = json.loads(raw_data)
        if isinstance(data_json, dict) and "data" in data_json:
            values = data_json["data"]
        elif isinstance(data_json, dict):
            values = list(data_json.values())
        elif isinstance(data_json, list):
            values = data_json
        else:
            values = data_json

        values = np.array(values).astype(float)
        pred_data = datamod.predict_dataloader(data=pd.DataFrame(values, columns=["Close"]))

        result, _ = mod(pred_data)
        result = scaler.inverse_transform(result.detach().numpy())

        return {"forecast": result.tolist()}
    except Exception as e:
        logging.exception("run() error: %s", str(e))
        return {"error": str(e)}

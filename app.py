"""
Traffic Congestion Prediction — Flask Web App
==============================================
Three pages: Home, Dashboard, Predict.
Loads the trained Keras model at startup and serves predictions.

Usage:
    python app.py
"""

import os, glob
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "models", "saved")

# ── Load trained model & artifacts once at startup ───────────────
model          = load_model(os.path.join(SAVE_DIR, "model.keras"))
scaler         = joblib.load(os.path.join(SAVE_DIR, "scaler.pkl"))
encoders       = joblib.load(os.path.join(SAVE_DIR, "label_encoders.pkl"))
feature_names  = joblib.load(os.path.join(SAVE_DIR, "feature_names.pkl"))
target_encoder = encoders["__target__"]

# ── Signal color mapping ────────────────────────────────────────
SIGNAL_COLORS = {
    "low": "green", "normal": "yellow", "high": "red", "heavy": "red",
}


def load_dataset():
    """Load the CSV from data/ for the dashboard."""
    csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    return pd.read_csv(csvs[0]) if csvs else None


# ── Routes ───────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
def dashboard():
    df = load_dataset()
    if df is None:
        return render_template("dashboard.html", error="No dataset found in data/ folder.")

    preview  = df.head(10).to_html(classes="table table-striped table-hover table-sm", index=False)
    shape    = df.shape
    missing  = df.isnull().sum().to_dict()
    dtypes   = df.dtypes.astype(str).to_dict()

    plot_dir = os.path.join("static", "plots")
    plots = {
        "correlation":  os.path.exists(os.path.join(plot_dir, "correlation_heatmap.png")),
        "distribution": os.path.exists(os.path.join(plot_dir, "congestion_distribution.png")),
        "vehicle":      os.path.exists(os.path.join(plot_dir, "vehicle_counts.png")),
    }

    return render_template(
        "dashboard.html",
        preview=preview, shape=shape,
        missing=missing, dtypes=dtypes, plots=plots,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction   = None
    signal_color = None

    if request.method == "POST":
        try:
            # Read the 6 features in the exact training order
            values = [
                float(request.form.get("Hour", 0)),
                float(request.form.get("DayOfWeek", 0)),
                float(request.form.get("Month", 1)),
                float(request.form.get("IsWeekend", 0)),
                float(request.form.get("Junction", 1)),
                float(request.form.get("Vehicles", 0)),
            ]

            # Scale and predict
            X = scaler.transform([values])
            proba = model.predict(X, verbose=0)
            pred_index = int(np.argmax(proba))
            prediction = str(target_encoder.inverse_transform([pred_index])[0])
            signal_color = SIGNAL_COLORS.get(prediction.strip().lower(), "yellow")

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template(
        "predict.html",
        prediction=prediction,
        signal_color=signal_color,
    )


if __name__ == "__main__":
    app.run(debug=True)

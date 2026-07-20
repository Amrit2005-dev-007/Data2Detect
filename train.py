"""
Traffic Congestion Prediction — Model Training Script
======================================================
Loads the traffic dataset, engineers features from DateTime,
trains a fast Keras neural network, and saves all artifacts.

Usage:
    python train.py
"""

import os, glob, joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
SAVE_DIR  = os.path.join(BASE_DIR, "models", "saved")
PLOT_DIR  = os.path.join(BASE_DIR, "static", "plots")


# ── Step 1: Load Data ───────────────────────────────────────────
def load_data():
    """Auto-detect the first CSV file in data/ folder."""
    csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csvs:
        raise FileNotFoundError("No CSV found in data/ folder.")
    print(f"[INFO] Dataset: {csvs[0]}")
    return pd.read_csv(csvs[0])


# ── Step 2: Feature Engineering & Preprocessing ─────────────────
def preprocess(df):
    """Engineer time features, create congestion labels, scale."""

    # Parse DateTime and extract useful features
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["Hour"]       = df["DateTime"].dt.hour
    df["DayOfWeek"]  = df["DateTime"].dt.dayofweek    # 0=Mon, 6=Sun
    df["Month"]      = df["DateTime"].dt.month
    df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)

    # Create congestion level from vehicle count using quartiles
    # This turns a regression problem into a 4-class classification
    q1 = df["Vehicles"].quantile(0.25)
    q2 = df["Vehicles"].quantile(0.50)
    q3 = df["Vehicles"].quantile(0.75)

    def label(v):
        if v <= q1:   return "low"
        elif v <= q2: return "normal"
        elif v <= q3: return "high"
        else:         return "heavy"

    df["Traffic Situation"] = df["Vehicles"].apply(label)

    # Select features for training
    feature_cols = ["Hour", "DayOfWeek", "Month", "IsWeekend", "Junction", "Vehicles"]
    target_col   = "Traffic Situation"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Fill any missing values
    X = X.fillna(0)

    # Encode target
    label_encoders = {}
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    label_encoders["__target__"] = target_le

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, scaler, label_encoders, feature_cols, df


# ── Step 3: Generate Dashboard Plots ────────────────────────────
def generate_plots(df):
    """Create and save visualizations for the dashboard."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Plot 1 — Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_cols = ["Hour", "DayOfWeek", "Month", "IsWeekend", "Junction", "Vehicles"]
    cols = [c for c in numeric_cols if c in df.columns]
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()

    # Plot 2 — Congestion distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["low", "normal", "high", "heavy"]
    counts = df["Traffic Situation"].value_counts().reindex(order)
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    counts.plot(kind="bar", color=colors, edgecolor="black", ax=ax)
    ax.set_title("Traffic Situation Distribution")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "congestion_distribution.png"), dpi=120)
    plt.close()

    # Plot 3 — Vehicles by hour
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby("Hour")["Vehicles"].mean().plot(kind="bar", color="#3498db", ax=ax)
    ax.set_title("Average Vehicle Count by Hour")
    ax.set_ylabel("Vehicles")
    ax.set_xlabel("Hour")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "vehicle_counts.png"), dpi=120)
    plt.close()

    print(f"[INFO] Plots saved to {PLOT_DIR}")


# ── Step 4: Build Model ─────────────────────────────────────────
def build_model(input_dim, num_classes):
    """Simple fast network: Input → 32 → 16 → Softmax."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Main Pipeline ────────────────────────────────────────────────
def main():
    # Load
    df = load_data()
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}\n")

    # Preprocess (creates time features + congestion labels)
    X, y, scaler, encoders, feature_names, df_processed = preprocess(df)
    num_classes = len(np.unique(y))
    print(f"[INFO] Features: {feature_names}")
    print(f"[INFO] Classes:  {list(encoders['__target__'].classes_)}")
    print(f"[INFO] Samples:  {len(X)}")

    # Generate plots
    generate_plots(df_processed)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build and train — FAST: only 10 epochs with larger batch
    model = build_model(X_train.shape[1], num_classes)
    model.summary()

    print("\n" + "=" * 50)
    print("  TRAINING (10 epochs — fast mode)")
    print("=" * 50)
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=256,
        verbose=1,
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'=' * 50}")
    print(f"  TEST RESULTS")
    print(f"{'=' * 50}")
    print(f"  Loss     : {loss:.4f}")
    print(f"  Accuracy : {accuracy:.4f}")

    # Save all artifacts
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save(os.path.join(SAVE_DIR, "model.keras"))
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(SAVE_DIR, "label_encoders.pkl"))
    joblib.dump(feature_names, os.path.join(SAVE_DIR, "feature_names.pkl"))

    print(f"\n[SAVED] model.keras")
    print(f"[SAVED] scaler.pkl")
    print(f"[SAVED] label_encoders.pkl")
    print(f"[SAVED] feature_names.pkl")
    print("\nDone!")


if __name__ == "__main__":
    main()

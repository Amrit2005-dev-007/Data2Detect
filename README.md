# 🚦 Data2Detect — Traffic Congestion Prediction System

A deep learning web application that predicts traffic congestion levels using a **Keras neural network** served through a **Flask** web interface.

Built as a portfolio project to demonstrate end-to-end machine learning — from data preprocessing to model training to web deployment.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web_App-green?logo=flask)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [How It Works](#how-it-works)
- [Future Improvements](#future-improvements)

---

## Overview

This project trains a simple feed-forward neural network on a traffic dataset to classify congestion into four levels: **Low**, **Normal**, **High**, and **Heavy**. The trained model is served via a Flask web app where users can input traffic features and get instant predictions displayed as a traffic signal.

---

## Features

- ✅ Automatic dataset preprocessing (missing values, encoding, scaling)
- ✅ Keras Sequential neural network (Dense → Dense → Softmax)
- ✅ Training script with evaluation metrics
- ✅ Flask web app with three pages:
  - **Home** — Project overview
  - **Dashboard** — Dataset stats, preview, and visualizations
  - **Predict** — Input form with traffic signal output
- ✅ Pre-generated visualizations (Matplotlib & Seaborn)
- ✅ Model and scaler saved for reuse

---

## Tech Stack

| Component        | Technology                |
|------------------|---------------------------|
| Language         | Python 3.10+              |
| ML Framework     | TensorFlow / Keras        |
| Data Processing  | Pandas, NumPy, Scikit-learn |
| Visualization    | Matplotlib, Seaborn       |
| Web Framework    | Flask                     |
| Frontend         | Bootstrap 5               |

---

## Project Structure

```
Data2Detect/
├── data/                       # Place your CSV dataset here
│   └── traffic.csv
├── models/
│   └── saved/                  # Trained model & artifacts
│       ├── model.keras
│       ├── scaler.pkl
│       ├── label_encoders.pkl
│       └── feature_names.pkl
├── static/
│   ├── css/
│   │   └── style.css           # Custom styles
│   └── plots/                  # Generated visualizations
│       ├── correlation_heatmap.png
│       ├── congestion_distribution.png
│       └── vehicle_counts.png
├── templates/
│   ├── base.html               # Base layout (Bootstrap)
│   ├── home.html               # Home page
│   ├── dashboard.html          # Dashboard page
│   └── predict.html            # Prediction page
├── app.py                      # Flask application
├── train.py                    # Model training script
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Data2Detect.git
cd Data2Detect
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the [Traffic Prediction Dataset](https://www.kaggle.com/datasets/hasanbasriakcay/traffic-prediction-dataset) from Kaggle and place the CSV file inside the `data/` folder.

### 5. Train the model

```bash
python train.py
```

This will:
- Preprocess the dataset
- Train the neural network
- Save the model, scaler, and encoders to `models/saved/`
- Generate plots to `static/plots/`

### 6. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## Usage

1. **Home** — Read about the project
2. **Dashboard** — Explore dataset shape, missing values, column types, and visualizations
3. **Predict** — Fill in traffic features (day, vehicle counts) and get a congestion prediction with a traffic signal indicator

---

## Screenshots

> Add your screenshots here after running the app.

| Home Page | Dashboard | Prediction |
|-----------|-----------|------------|
| ![Home](screenshots/home.png) | ![Dashboard](screenshots/dashboard.png) | ![Predict](screenshots/predict.png) |

---

## How It Works

```
CSV Dataset → Preprocessing → Train/Test Split → Keras Model → Saved Artifacts
                                                                     ↓
                                                      Flask App → User Input → Prediction
                                                                     ↓
                                                           Traffic Signal Output
```

**Preprocessing:**
- Drop `Date`, `Time`, `Total` columns
- Fill missing values (median / mode)
- Encode categorical columns with `LabelEncoder`
- Scale features with `StandardScaler`

**Model Architecture:**
```
Input → Dense(32, ReLU) → Dense(16, ReLU) → Dense(4, Softmax)
```

**Output Mapping:**
| Prediction | Signal Color |
|------------|-------------|
| Low        | 🟢 Green    |
| Normal     | 🟡 Yellow   |
| High       | 🔴 Red      |
| Heavy      | 🔴 Red      |

---

## Future Improvements

- [ ] Add real-time traffic data integration via API
- [ ] Use LSTM/GRU for time-series prediction
- [ ] Add model comparison (Random Forest, XGBoost vs Neural Network)
- [ ] Deploy on Heroku / Render / Railway
- [ ] Add user authentication and prediction history
- [ ] Implement interactive Plotly charts on the dashboard
- [ ] Add unit tests for preprocessing and prediction

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ for learning and portfolio purposes.
</p>

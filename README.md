# Data2Detect

Traffic congestion prediction app using Flask and Keras. Feed it traffic data and it tells you whether congestion is **Low**, **Normal**, **High**, or **Heavy** — with a traffic signal on the web page.

Built this as a portfolio project to go through the full ML workflow — data preprocessing, model training, and serving predictions through a simple web app.

---

## What it does

- Reads a traffic CSV dataset and pulls out time-based features (hour, day, month, etc.)
- Trains a small Keras neural network (Dense 32 → 16 → 4 softmax)
- Saves the trained model, scaler, and encoders
- Runs a Flask web app with three pages:
  - **Home** — project overview
  - **Dashboard** — dataset stats, preview table, correlation heatmap, distribution plots
  - **Predict** — input form → prediction → traffic light indicator

---

## Project structure

```
Data2Detect/
├── data/                    # dataset goes here
│   └── traffic.csv
├── models/saved/            # created after training
├── static/
│   ├── css/style.css
│   └── plots/               # created after training
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── dashboard.html
│   └── predict.html
├── app.py                   # flask app
├── train.py                 # training script
├── requirements.txt
└── README.md
```

---

## Setup

Needs Python 3.10+ and the packages in `requirements.txt`. I used [this Kaggle dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset) — download the CSV and put it in `data/`.

```bash
pip install -r requirements.txt
python train.py        # trains the model, saves artifacts, generates plots
python app.py          # starts the flask app on localhost:5000
```

Training completed successfully on the traffic dataset. The model achieved approximately **91.47% test accuracy** and is used by the Flask application for real-time traffic congestion prediction.

---






## Tech stack

Python, Flask, TensorFlow/Keras, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Bootstrap 5

---

## Ideas for later

- Try LSTM/GRU since this is really time-series data
- Swap static matplotlib plots for interactive Plotly charts
- Compare against simpler models like Random Forest or XGBoost
- Deploy somewhere (Render, Railway)

---

MIT License

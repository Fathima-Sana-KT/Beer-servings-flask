from pathlib import Path
import json
import urllib.request
import subprocess
import sys
import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "drinks.csv"
MODEL_PATH = BASE_DIR / "best_model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"

app = Flask(__name__)

def ensure_dataset():
    if not DATA_PATH.exists():
        url = "https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/drinks.csv"
        try:
            urllib.request.urlretrieve(url, DATA_PATH)
        except Exception as exc:
            raise RuntimeError("Could not download drinks.csv automatically. Check hosting network access.") from exc

def ensure_model():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        result = subprocess.run([sys.executable, str(BASE_DIR / "train_model.py")],
                                cwd=BASE_DIR, capture_output=True, text=True, timeout=240)
        if result.returncode != 0:
            raise RuntimeError("Model training failed: " + result.stderr[-1200:])

def load_data():
    ensure_dataset()
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    required = ["country", "beer_servings", "spirit_servings", "wine_servings",
                "total_litres_of_pure_alcohol", "continent"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df

def make_charts(df):
    continent = (df.groupby("continent", dropna=False)["total_litres_of_pure_alcohol"]
                   .mean().reset_index().sort_values("total_litres_of_pure_alcohol", ascending=False))
    fig1 = px.bar(continent, x="continent", y="total_litres_of_pure_alcohol",
                  title="Average pure alcohol litres by continent",
                  labels={"continent": "Continent", "total_litres_of_pure_alcohol": "Litres"})
    fig1.update_layout(template="plotly_white", margin=dict(l=15, r=15, t=55, b=15))
    fig2 = px.scatter(df, x="beer_servings", y="total_litres_of_pure_alcohol",
                      color="continent", hover_name="country",
                      title="Beer servings vs total pure alcohol",
                      labels={"beer_servings": "Beer servings", "total_litres_of_pure_alcohol": "Litres"})
    fig2.update_layout(template="plotly_white", margin=dict(l=15, r=15, t=55, b=15))
    return fig1.to_html(full_html=False, include_plotlyjs="cdn"), fig2.to_html(full_html=False, include_plotlyjs=False)

@app.route("/", methods=["GET", "POST"])
def index():
    df = load_data()
    ensure_model()
    model = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text())
    chart1, chart2 = make_charts(df)
    countries = sorted(df["country"].dropna().unique().tolist())
    continents = sorted(df["continent"].dropna().unique().tolist())
    prediction = None
    error = None

    if request.method == "POST":
        try:
            country = request.form["country"]
            continent = request.form["continent"]
            values = {
                "country": country,
                "beer_servings": float(request.form["beer_servings"]),
                "spirit_servings": float(request.form["spirit_servings"]),
                "wine_servings": float(request.form["wine_servings"]),
                "continent": continent
            }
            if any(values[k] < 0 for k in ["beer_servings", "spirit_servings", "wine_servings"]):
                raise ValueError("Servings must be zero or a positive number.")
            row = pd.DataFrame([values])
            prediction = max(0.0, float(model.predict(row)[0]))
        except (KeyError, ValueError, TypeError) as exc:
            error = str(exc) or "Please enter valid input values."

    return render_template("index.html", countries=countries, continents=continents,
                           chart1=chart1, chart2=chart2, prediction=prediction,
                           error=error, metrics=metrics,
                           form=request.form if request.method == "POST" else {})

if __name__ == "__main__":
    app.run(debug=True)

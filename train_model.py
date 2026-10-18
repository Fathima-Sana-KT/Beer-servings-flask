from pathlib import Path
import json
import urllib.request
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "drinks.csv"
MODEL_PATH = BASE_DIR / "best_model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"

if not DATA_PATH.exists():
    url = "https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/drinks.csv"
    print("Downloading the Beer Servings dataset...")
    urllib.request.urlretrieve(url, DATA_PATH)

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]
features = ["country", "beer_servings", "spirit_servings", "wine_servings", "continent"]
target = "total_litres_of_pure_alcohol"
df = df[features + [target]].dropna()
X, y = df[features], df[target]

categorical = ["country", "continent"]
numeric = ["beer_servings", "spirit_servings", "wine_servings"]
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numeric", "passthrough", numeric)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Random Forest": (
        RandomForestRegressor(random_state=42),
        {"regressor__n_estimators": [100, 200],
         "regressor__max_depth": [None, 5, 10],
         "regressor__min_samples_split": [2, 5]}
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=42),
        {"regressor__n_estimators": [100, 200],
         "regressor__learning_rate": [0.03, 0.1],
         "regressor__max_depth": [2, 3]}
    ),
    "Extra Trees": (
        ExtraTreesRegressor(random_state=42),
        {"regressor__n_estimators": [100, 200],
         "regressor__max_depth": [None, 5, 10],
         "regressor__min_samples_split": [2, 5]}
    )
}

results = []
best_name, best_score, best_search = None, float("-inf"), None

for name, (estimator, grid) in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])
    search = GridSearchCV(pipe, grid, cv=5, scoring="r2", n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    preds = search.predict(X_test)
    score = float(r2_score(y_test, preds))
    results.append({
        "model": name,
        "evaluation_r2": score,
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "best_params": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in search.best_params_.items()}
    })
    print(f"{name}: evaluation R²={score:.4f}; best params={search.best_params_}")
    if score > best_score:
        best_name, best_score, best_search = name, score, search

joblib.dump(best_search.best_estimator_, MODEL_PATH)
metrics = {
    "best_model": best_name,
    "best_evaluation_r2": best_score,
    "models": sorted(results, key=lambda item: item["evaluation_r2"], reverse=True),
    "rows_used": int(len(df)),
    "test_size": 0.2,
    "random_state": 42
}
METRICS_PATH.write_text(json.dumps(metrics, indent=2))
print(f"\nSelected model: {best_name} (evaluation R²={best_score:.4f})")
print(f"Saved model to {MODEL_PATH}")
print(f"Saved metrics to {METRICS_PATH}")

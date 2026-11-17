"""
03a_regression.py
Predict SNOTEL snow depth with a random forest and compare it to simpler
models using leave-one-station-out cross-validation: each station is
predicted by a model trained only on the other stations.

Models:
    elev_lr     linear regression on elevation only (baseline)
    mlr         multiple linear regression on all features (baseline)
    rf_noradar  random forest on terrain + climate only (tests what radar adds)
    rf_full     random forest on all features (the main model)

Outputs: data/processed/cv_predictions.csv  (out-of-station predictions, every model)
         data/processed/cv_metrics.csv      (RMSE, MAE, bias, R2 per model)
         models/rf_depth.joblib             (rf_full trained on all stations, for mapping)
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
import config as c

c.make_dirs()


def make_rf():
    """Random forest settings shared by both RF models."""
    return RandomForestRegressor(
        n_estimators=500,       # number of trees
        min_samples_leaf=5,     # each leaf averages at least 5 observations (limits overfitting)
        random_state=42,        # same result every run
        n_jobs=-1,              # use all CPU cores
    )


def score(obs, pred):
    """Error metrics for a set of predictions (cm, except R2)."""
    return {
        "rmse": np.sqrt(mean_squared_error(obs, pred)),
        "mae": mean_absolute_error(obs, pred),
        "bias": np.mean(pred - obs),    # + = overpredicts, - = underpredicts
        "r2": r2_score(obs, pred),
    }


# ---------------- Data ----------------
df = pd.read_csv(c.processed / "model_table.csv", parse_dates=["date"])
y = df["depth_cm"]
groups = df["id"]   # leave out one station at a time
print(f"{len(df)} observations from {groups.nunique()} stations")

models = {
    "elev_lr": (LinearRegression(), ["elev"]),
    "mlr": (LinearRegression(), c.features),
    "rf_noradar": (make_rf(), c.terrain_feats + c.climate_feats),
    "rf_full": (make_rf(), c.features),
}

# ---------------- Leave-one-station-out cross-validation ----------------
cv = LeaveOneGroupOut()
preds = df[["id", "name", "date", "depth_cm"]].copy()
results = []
for name, (model, cols) in models.items():
    preds[name] = cross_val_predict(model, df[cols], y, groups=groups, cv=cv)
    results.append({"model": name, "n_features": len(cols), **score(y, preds[name])})
    print(f"  {name} done")

metrics = pd.DataFrame(results)
print("\nLeave-one-station-out results (cm):")
print(metrics.round(2).to_string(index=False))

gain = metrics.set_index("model")["rmse"]
print(f"\nRadar effect: RMSE {gain['rf_noradar']:.1f} cm without radar -> "
      f"{gain['rf_full']:.1f} cm with radar ({gain['rf_noradar'] - gain['rf_full']:+.1f} cm)")

preds.round(2).to_csv(c.processed / "cv_predictions.csv", index=False)
metrics.round(3).to_csv(c.processed / "cv_metrics.csv", index=False)

# ---------------- Final model for mapping ----------------
# Train rf_full on ALL stations and save it with its feature list
final = make_rf().fit(df[c.features], y)
joblib.dump({"model": final, "features": c.features}, c.models_dir / "rf_depth.joblib")
print(f"\nSaved final model to models/rf_depth.joblib")
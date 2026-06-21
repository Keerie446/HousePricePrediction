"""Train a regression pipeline for house price prediction.

Usage:
    python -m src.train --data data/housing.csv --out models/model.joblib
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def build_pipeline(df, target_col="price"):
    X = df.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=[object, "category"]).columns.tolist()

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # Create OneHotEncoder with a parameter compatible across scikit-learn versions
    from sklearn import __version__ as _skl_ver
    # Avoid distutils; compare major/minor version numbers safely
    try:
        parts = _skl_ver.split(".")
        major, minor = int(parts[0]), int(parts[1])
    except Exception:
        major, minor = 0, 0
    if (major, minor) >= (1, 2):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ], remainder="drop")

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def load_data(path: Path, target_col="price"):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    return df


def main(args):
    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path, target_col=args.target)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline(df, target_col=args.target)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    # Compute RMSE in a way compatible with older sklearn versions
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")

    joblib.dump(pipeline, out_path)
    print(f"Saved model pipeline to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--out", required=True, help="Output path for saved model (joblib)")
    parser.add_argument("--target", default="price", help="Name of target column")
    args = parser.parse_args()
    main(args)

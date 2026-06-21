"""Load a saved pipeline and make predictions on input CSV.

Usage:
    python -m src.predict --model models/model.joblib --input data/sample_input.csv
"""
from pathlib import Path
import argparse
import pandas as pd
import joblib


def main(args):
    model_path = Path(args.model)
    input_path = Path(args.input)

    pipeline = joblib.load(model_path)
    df = pd.read_csv(input_path)
    preds = pipeline.predict(df)
    out = df.copy()
    out["predicted_price"] = preds
    print(out.head())
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"Wrote predictions to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved model joblib")
    parser.add_argument("--input", required=True, help="CSV with input features")
    parser.add_argument("--out", help="Optional output CSV for predictions")
    args = parser.parse_args()
    main(args)

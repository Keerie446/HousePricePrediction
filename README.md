# House Price Prediction — Week 1

Minimal ML project scaffold for the internship task: clean data, train a regression model, evaluate, and save the trained pipeline.

Getting started

1. Place your dataset CSV at `data/housing.csv` (header row expected).
2. Create/activate the virtualenv in this folder and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training

```bash
python -m src.train --data data/housing.csv --out models/model.joblib
```

Prediction

```bash
python -m src.predict --model models/model.joblib --input data/sample_input.csv
```

If you don't yet have a dataset, provide a CSV with numeric columns for common house features (size, rooms, age, etc.) and a `price` column as the target.

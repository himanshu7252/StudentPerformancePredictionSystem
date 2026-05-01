from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "final_score"


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned = cleaned.drop_duplicates(subset=["student_id"])
    return cleaned


def build_pipeline(feature_frame: pd.DataFrame) -> Pipeline:
    numeric_features = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = feature_frame.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_model(data: pd.DataFrame, target_column: str = TARGET_COLUMN):
    cleaned = clean_data(data)
    features = cleaned.drop(columns=[target_column, "performance_level", "student_id"], errors="ignore")
    target = cleaned[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions)

    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
        "cleaned_data": cleaned,
    }


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def predict_single_student(pipeline: Pipeline, student_input: dict[str, Any]) -> float:
    input_frame = pd.DataFrame([student_input])
    prediction = pipeline.predict(input_frame)[0]
    return float(prediction)


def save_model(pipeline: Pipeline, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return output_path


def load_model(model_path: str | Path) -> Pipeline:
    return joblib.load(model_path)

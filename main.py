from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, IMAGES_DIR, MODELS_DIR, OUTPUTS_DIR
from src.data_generation import generate_student_data, save_dataset
from src.modeling import predict_single_student, save_model, train_model
from src.visualization import save_eda_plots, save_model_plots


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Student Performance Prediction System")
    parser.add_argument("--samples", type=int, default=2000, help="Number of synthetic student records to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser


def ensure_directories() -> None:
    for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def create_demo_student() -> dict[str, object]:
    return {
        "gender": "Female",
        "school_type": "Private",
        "study_environment": "Quiet",
        "study_hours_per_week": 12.0,
        "attendance_percentage": 91.0,
        "previous_score": 78.0,
        "sleep_hours": 7.5,
        "assignments_completed": 18,
        "mock_tests_attempted": 7,
        "participation_score": 8.2,
        "parental_support_level": 4,
        "internet_access": 1,
        "stress_level": 3.0,
        "screen_time_hours": 2.5,
        "commute_time_mins": 20.0,
        "extracurricular_hours": 4.0,
        "tuition_support": 1,
        "student_id": 999999,
    }


def main() -> None:
    args = build_argument_parser().parse_args()
    ensure_directories()

    print("Generating synthetic student dataset...")
    dataset = generate_student_data(n_samples=args.samples, random_state=args.seed)
    dataset_path = save_dataset(dataset, DEFAULT_DATASET_PATH)
    print(f"Saved dataset to: {dataset_path}")

    print("Training model...")
    training_result = train_model(dataset)
    pipeline = training_result["pipeline"]
    metrics = training_result["metrics"]

    print("Saving model and evaluation artifacts...")
    model_path = save_model(pipeline, DEFAULT_MODEL_PATH)
    eda_files = save_eda_plots(training_result["cleaned_data"], IMAGES_DIR)
    model_files = save_model_plots(
        pipeline,
        training_result["X_test"],
        training_result["y_test"],
        training_result["predictions"],
        IMAGES_DIR,
    )

    prediction_frame = training_result["X_test"].copy()
    prediction_frame["actual_final_score"] = training_result["y_test"].values
    prediction_frame["predicted_final_score"] = training_result["predictions"]
    prediction_path = OUTPUTS_DIR / "sample_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    metrics_path = OUTPUTS_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    demo_student = create_demo_student()
    demo_score = predict_single_student(pipeline, demo_student)
    demo_band = "At Risk" if demo_score < 50 else "Safe"

    summary_path = OUTPUTS_DIR / "run_summary.txt"
    summary_content = [
        "Student Performance Prediction System Run Summary",
        f"Dataset path: {dataset_path}",
        f"Model path: {model_path}",
        f"Evaluation metrics: {json.dumps(metrics, indent=2)}",
        f"Demo prediction score: {demo_score:.2f}",
        f"Demo prediction band: {demo_band}",
        f"EDA plots: {[str(path) for path in eda_files]}",
        f"Model plots: {[str(path) for path in model_files]}",
    ]
    summary_path.write_text("\n".join(summary_content), encoding="utf-8")

    print("\nProject completed successfully.")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R2: {metrics['r2']:.3f}")
    print(f"Demo student predicted score: {demo_score:.2f}")
    print(f"Artifacts saved in: {OUTPUTS_DIR} and {IMAGES_DIR}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _clip(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return series.clip(lower=lower, upper=upper)


def generate_student_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic student performance dataset."""
    rng = np.random.default_rng(random_state)

    study_hours = rng.gamma(shape=2.2, scale=2.2, size=n_samples)
    study_hours = np.clip(study_hours, 0, 18)
    attendance = rng.normal(loc=82, scale=12, size=n_samples)
    attendance = np.clip(attendance, 35, 100)
    previous_score = rng.normal(loc=68, scale=15, size=n_samples)
    previous_score = np.clip(previous_score, 0, 100)
    sleep_hours = rng.normal(loc=7.0, scale=1.1, size=n_samples)
    sleep_hours = np.clip(sleep_hours, 4, 10)
    assignments_completed = rng.integers(0, 21, size=n_samples)
    mock_tests_attempted = rng.integers(0, 11, size=n_samples)
    participation_score = rng.uniform(0, 10, size=n_samples)
    parental_support_level = rng.integers(0, 6, size=n_samples)
    internet_access = rng.integers(0, 2, size=n_samples)
    stress_level = rng.uniform(0, 10, size=n_samples)
    screen_time_hours = np.clip(rng.normal(loc=3.5, scale=1.6, size=n_samples), 0, 10)
    commute_time_mins = np.clip(rng.normal(loc=28, scale=18, size=n_samples), 0, 120)
    extracurricular_hours = np.clip(rng.normal(loc=4, scale=2.5, size=n_samples), 0, 15)
    tuition_support = rng.integers(0, 2, size=n_samples)
    gender = rng.choice(["Female", "Male", "Other"], size=n_samples, p=[0.48, 0.49, 0.03])
    school_type = rng.choice(["Government", "Private", "Semi-Private"], size=n_samples, p=[0.45, 0.35, 0.20])
    study_environment = rng.choice(["Quiet", "Average", "Noisy"], size=n_samples, p=[0.42, 0.40, 0.18])

    noise = rng.normal(0, 5.5, size=n_samples)
    final_score = (
        16
        + 0.85 * study_hours
        + 0.18 * attendance
        + 0.34 * previous_score
        + 0.75 * assignments_completed
        + 1.15 * mock_tests_attempted
        + 1.25 * participation_score
        + 1.75 * parental_support_level
        + 1.4 * internet_access
        + 0.55 * sleep_hours
        + 0.25 * extracurricular_hours
        + 0.7 * tuition_support
        - 1.35 * stress_level
        - 0.95 * screen_time_hours
        - 0.03 * commute_time_mins
        + noise
    )
    final_score = np.clip(final_score, 0, 100)

    performance_level = pd.cut(
        final_score,
        bins=[-0.1, 49.9, 74.9, 100.0],
        labels=["Low", "Medium", "High"],
    )

    data = pd.DataFrame(
        {
            "student_id": np.arange(1, n_samples + 1),
            "gender": gender,
            "school_type": school_type,
            "study_environment": study_environment,
            "study_hours_per_week": np.round(study_hours, 2),
            "attendance_percentage": np.round(attendance, 2),
            "previous_score": np.round(previous_score, 2),
            "sleep_hours": np.round(sleep_hours, 2),
            "assignments_completed": assignments_completed,
            "mock_tests_attempted": mock_tests_attempted,
            "participation_score": np.round(participation_score, 2),
            "parental_support_level": parental_support_level,
            "internet_access": internet_access,
            "stress_level": np.round(stress_level, 2),
            "screen_time_hours": np.round(screen_time_hours, 2),
            "commute_time_mins": np.round(commute_time_mins, 2),
            "extracurricular_hours": np.round(extracurricular_hours, 2),
            "tuition_support": tuition_support,
            "final_score": np.round(final_score, 2),
            "performance_level": performance_level.astype(str),
        }
    )

    missing_mask = rng.random(data.shape) < 0.03
    protected_columns = {"student_id", "final_score", "performance_level"}
    for column_index, column_name in enumerate(data.columns):
        if column_name in protected_columns:
            continue
        data.loc[missing_mask[:, column_index], column_name] = np.nan

    return data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def save_dataset(data: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path

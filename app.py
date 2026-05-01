from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from main import create_demo_student, ensure_directories
from src.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, IMAGES_DIR, OUTPUTS_DIR
from src.data_generation import generate_student_data, save_dataset
from src.modeling import predict_single_student, save_model, train_model
from src.visualization import save_eda_plots, save_model_plots


st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(90, 126, 255, 0.18), transparent 30%),
            radial-gradient(circle at top right, rgba(255, 107, 107, 0.16), transparent 28%),
            linear-gradient(180deg, #08111f 0%, #0f172a 100%);
        color: #f8fafc;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    h1, h2, h3, h4, h5, h6, p, li, label {
        color: #f8fafc !important;
    }
    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.6));
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 18px 50px rgba(2, 6, 23, 0.35);
    }
    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0 12px 36px rgba(2, 6, 23, 0.26);
    }
    .soft-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.14);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }
    .stDataFrame, .stPlotlyChart, .stImage {
        border-radius: 18px;
        overflow: hidden;
    }
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def load_or_train_pipeline(samples: int, seed: int):
    ensure_directories()
    if DEFAULT_MODEL_PATH.exists() and DEFAULT_DATASET_PATH.exists():
        from src.modeling import load_model

        model = load_model(DEFAULT_MODEL_PATH)
        data = pd.read_csv(DEFAULT_DATASET_PATH)
        return model, data, None

    data = generate_student_data(n_samples=samples, random_state=seed)
    save_dataset(data, DEFAULT_DATASET_PATH)
    training_result = train_model(data)
    model = training_result["pipeline"]
    save_model(model, DEFAULT_MODEL_PATH)
    save_eda_plots(training_result["cleaned_data"], IMAGES_DIR)
    save_model_plots(
        model,
        training_result["X_test"],
        training_result["y_test"],
        training_result["predictions"],
        IMAGES_DIR,
    )
    return model, data, training_result


def performance_band(score: float) -> str:
    if score < 50:
        return "At Risk"
    if score < 75:
        return "Moderate"
    return "Excellent"


def insight_message(score: float) -> str:
    if score < 50:
        return "Focus on attendance, weekly study hours, and stress reduction."
    if score < 75:
        return "The student is progressing, but consistency and practice tests can improve results."
    return "The student is performing strongly and should continue the current learning pattern."


st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom: 0.25rem;">Student Performance Prediction System</h1>
        <p style="margin-top: 0; opacity: 0.9;">Modern analytics dashboard for predicting student outcomes using synthetic but realistic data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.title("Control Panel")
    st.caption("Generate data, train the model, and explore predictions.")
    sample_count = st.slider("Synthetic dataset size", min_value=500, max_value=5000, value=2000, step=250)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)
    regenerate = st.button("Generate / Refresh Model", use_container_width=True)
    st.divider()
    st.markdown("**Project files**")
    st.write(f"Dataset: `{DEFAULT_DATASET_PATH.name}`")
    st.write(f"Model: `{DEFAULT_MODEL_PATH.name}`")
    st.write(f"Outputs: `{OUTPUTS_DIR.name}`")


if regenerate or not DEFAULT_MODEL_PATH.exists() or not DEFAULT_DATASET_PATH.exists():
    with st.spinner("Preparing data and model..."):
        model, dataset, training_result = load_or_train_pipeline(sample_count, int(seed))
else:
    from src.modeling import load_model

    model = load_model(DEFAULT_MODEL_PATH)
    dataset = pd.read_csv(DEFAULT_DATASET_PATH)
    training_result = None


if training_result is None:
    if DEFAULT_DATASET_PATH.exists():
        st.success("Loaded the saved model and dataset. Use the panel below to make a prediction.")
else:
    metrics = training_result["metrics"]
    cols = st.columns(4)
    cols[0].metric("MAE", f"{metrics['mae']:.2f}")
    cols[1].metric("RMSE", f"{metrics['rmse']:.2f}")
    cols[2].metric("R2", f"{metrics['r2']:.3f}")
    cols[3].metric("Rows", f"{len(dataset):,}")


st.markdown("## Predict a Student Outcome")
left, right = st.columns([1.05, 1.2], gap="large")

with left:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Student Profile")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=0)
    school_type = st.selectbox("School type", ["Government", "Private", "Semi-Private"], index=1)
    study_environment = st.selectbox("Study environment", ["Quiet", "Average", "Noisy"], index=0)
    study_hours_per_week = st.slider("Study hours per week", 0.0, 20.0, 12.0, 0.5)
    attendance_percentage = st.slider("Attendance percentage", 0.0, 100.0, 91.0, 1.0)
    previous_score = st.slider("Previous score", 0.0, 100.0, 78.0, 1.0)
    sleep_hours = st.slider("Sleep hours", 0.0, 12.0, 7.5, 0.1)
    assignments_completed = st.slider("Assignments completed", 0, 20, 18, 1)
    mock_tests_attempted = st.slider("Mock tests attempted", 0, 10, 7, 1)
    participation_score = st.slider("Participation score", 0.0, 10.0, 8.2, 0.1)
    parental_support_level = st.slider("Parental support level", 0, 5, 4, 1)
    internet_access = st.selectbox("Internet access", [0, 1], index=1)
    stress_level = st.slider("Stress level", 0.0, 10.0, 3.0, 0.1)
    screen_time_hours = st.slider("Screen time hours", 0.0, 12.0, 2.5, 0.1)
    commute_time_mins = st.slider("Commute time (mins)", 0.0, 180.0, 20.0, 1.0)
    extracurricular_hours = st.slider("Extracurricular hours", 0.0, 20.0, 4.0, 0.5)
    tuition_support = st.selectbox("Tuition support", [0, 1], index=1)
    predict_button = st.button("Predict Performance", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

student_input = {
    "gender": gender,
    "school_type": school_type,
    "study_environment": study_environment,
    "study_hours_per_week": study_hours_per_week,
    "attendance_percentage": attendance_percentage,
    "previous_score": previous_score,
    "sleep_hours": sleep_hours,
    "assignments_completed": assignments_completed,
    "mock_tests_attempted": mock_tests_attempted,
    "participation_score": participation_score,
    "parental_support_level": parental_support_level,
    "internet_access": internet_access,
    "stress_level": stress_level,
    "screen_time_hours": screen_time_hours,
    "commute_time_mins": commute_time_mins,
    "extracurricular_hours": extracurricular_hours,
    "tuition_support": tuition_support,
    "student_id": 999999,
}

with right:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")
    if predict_button:
        predicted_score = predict_single_student(model, student_input)
        band = performance_band(predicted_score)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Score", f"{predicted_score:.2f}")
        c2.metric("Risk Band", band)
        c3.metric("Insight", "Stable" if band != "At Risk" else "Needs support")
        st.progress(min(max(predicted_score / 100.0, 0.0), 1.0))
        st.markdown(f"### {insight_message(predicted_score)}")
        st.markdown("#### Input Snapshot")
        st.dataframe(pd.DataFrame([student_input]).drop(columns=["student_id"]), use_container_width=True)
    else:
        st.info("Adjust the student profile on the left and click Predict Performance.")
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("## Dataset Explorer")
tab1, tab2, tab3 = st.tabs(["Preview", "Charts", "Generated Artifacts"])

with tab1:
    st.dataframe(dataset.head(20), use_container_width=True)

with tab2:
    image_paths = sorted(Path(IMAGES_DIR).glob("*.png"))
    if image_paths:
        cols = st.columns(2)
        for index, image_path in enumerate(image_paths):
            with cols[index % 2]:
                st.image(str(image_path), caption=image_path.name, use_container_width=True)
    else:
        st.info("Run a training cycle to generate charts.")

with tab3:
    output_files = sorted(Path(OUTPUTS_DIR).glob("*"))
    if output_files:
        for file_path in output_files:
            st.write(f"- {file_path.name}")
    else:
        st.info("No output files found yet.")

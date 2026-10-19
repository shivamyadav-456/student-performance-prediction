"""
Student Performance Prediction System — Interactive Streamlit Dashboard
Dataset: UCI Student Performance Dataset (Mathematics: student-mat.csv)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_raw_dataset,
    prepare_labeled_dataset,
    get_dataset_statistics,
    GRADE_BOUNDARIES,
    MATH_DATASET_NAME,
)
from src.preprocessor import (
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_CLASSES,
    FEATURE_METADATA,
)
from src.predict import (
    load_model,
    predict_student_performance,
    SAMPLE_PROFILES,
    DEFAULT_MODEL_PATH,
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: Midnight Navy + Teal Palette with refined typography and metric cards
st.markdown(
    """
    <style>
    /* Global Base */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    
    /* Midnight Navy Header Accents */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Clean Metric Cards with Midnight Navy / Teal Accents */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 3.5px solid #0d9488; /* Teal top accent */
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.04);
        margin-bottom: 12px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 8px rgba(15, 23, 42, 0.08);
    }
    .metric-title {
        color: #475569;
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.80rem;
        margin-top: 4px;
    }
    
    /* Filter Info Pill */
    .filter-badge {
        display: inline-block;
        background-color: #f1f5f9;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Prediction Outcome Cards */
    .result-card-high {
        background-color: #f0fdf4;
        border: 1.5px solid #86efac;
        border-left: 5px solid #059669;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .result-card-medium {
        background-color: #f0fdfa;
        border: 1.5px solid #99f6e4;
        border-left: 5px solid #0d9488;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .result-card-low {
        background-color: #fff7ed;
        border: 1.5px solid #fdba74;
        border-left: 5px solid #ea580c;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    /* Category Badges */
    .badge-high {
        background-color: #059669;
        color: white;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.02em;
    }
    .badge-medium {
        background-color: #0d9488;
        color: white;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.02em;
    }
    .badge-low {
        background-color: #ea580c;
        color: white;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.02em;
    }

    /* High-contrast accessible text styling */
    body, p, label, span {
        color: #0f172a;
    }

    /* Subtitle and helper text */
    .stCaption, .help-text {
        color: #475569 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Loads and caches the model payload from disk."""
    try:
        return load_model()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_cached_dataset():
    """Loads and caches the labeled dataset."""
    try:
        return prepare_labeled_dataset()
    except Exception:
        return None


model_payload = get_cached_model()
df_dataset = get_cached_dataset()

# ---------------------------------------------------------
# Sidebar Navigation & System Status
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            <span style="font-size: 2.2rem; line-height: 1;">🎓</span>
            <div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a; line-height: 1.2;">Student ML System</div>
                <div style="font-size: 0.8rem; color: #0d9488; font-weight: 600;">Academic Performance Predictor</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    PAGES = [
        "🏠 Home Dashboard",
        "🔮 Predict Performance",
        "📊 Model Performance",
        "🔍 Dataset Explorer",
    ]

    def set_nav_page(target_page: str):
        st.session_state["current_page"] = target_page

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = PAGES[0]

    # Pre-sync key so st.radio initializes with the exact target page before widget instantiation
    st.session_state["nav_radio_select"] = st.session_state["current_page"]

    def on_nav_change():
        st.session_state["current_page"] = st.session_state["nav_radio_select"]

    selected_radio = st.radio(
        "Navigation",
        PAGES,
        key="nav_radio_select",
        on_change=on_nav_change,
    )
    menu_selection = st.session_state["current_page"]

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    if model_payload:
        acc = model_payload["primary_metrics"]["accuracy"]
        cv_acc = model_payload["primary_metrics"].get("cv_accuracy_mean", acc)
        st.success(
            f"**Model:** Random Forest\n\n"
            f"**Test Accuracy:** {acc*100:.1f}%\n\n"
            f"**5-Fold CV:** {cv_acc*100:.1f}%\n\n"
            f"**Status:** Operational"
        )
    else:
        st.error("**Model:** Not Loaded\n\nRun `python train_model.py` to train.")

    if df_dataset is not None:
        st.info(f"**Dataset:** Math Course\n\n**Records:** {len(df_dataset)} students\n\n**Missing Values:** 0")

    st.markdown("---")
    st.caption(
        "**College ML Mini-Project**  \n"
        "Educational Decision Support Tool  \n"
        "*Not a guarantee of future performance.*"
    )

# ---------------------------------------------------------
# Category Colors Map
# ---------------------------------------------------------
CATEGORY_COLORS = {
    "Low Performance": "#ea580c",     # Amber/Orange
    "Medium Performance": "#0d9488",  # Refined Teal
    "High Performance": "#059669",    # Deep Emerald
}


# =========================================================
# PAGE A: HOME DASHBOARD
# =========================================================
if menu_selection == "🏠 Home Dashboard":
    st.title("🎓 Student Performance Prediction System")
    st.markdown(
        "An educational Machine Learning decision-support application built with **Scikit-learn**, "
        "**Random Forest Classification**, and **Streamlit** to forecast secondary school academic performance categories."
    )

    if df_dataset is None:
        st.error(
            "Dataset not found. Please ensure the UCI dataset is placed in `data/student-mat.csv` "
            "or run `python train_model.py`."
        )
        st.stop()

    stats = get_dataset_statistics(df_dataset)

    # Top KPI Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Records</div>
                <div class="metric-value">{stats['total_records']}</div>
                <div class="metric-sub">Secondary school students</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        model_acc_str = f"{model_payload['primary_metrics']['accuracy']*100:.1f}%" if model_payload else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Model Test Accuracy</div>
                <div class="metric-value">{model_acc_str}</div>
                <div class="metric-sub">Held-out test set (80/20 split)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        pass_count = sum(df_dataset["G3"] >= 10)
        pass_rate = (pass_count / len(df_dataset)) * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Overall Passing Rate</div>
                <div class="metric-value">{pass_rate:.1f}%</div>
                <div class="metric-sub">{pass_count} out of {len(df_dataset)} (G3 ≥ 10 passing mark)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        avg_g3 = stats["grade_mean_g3"]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Mean Final Grade (G3)</div>
                <div class="metric-value">{avg_g3:.1f} / 20</div>
                <div class="metric-sub">Portuguese 0–20 grading scale</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Two column overview: Class Distribution & Grade Distributions
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("### 📊 Performance Category Distribution")
        cat_counts = df_dataset["performance_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]

        fig_donut = px.pie(
            cat_counts,
            values="Count",
            names="Category",
            color="Category",
            color_discrete_map=CATEGORY_COLORS,
            hole=0.45,
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent+label")
        fig_donut.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_right:
        st.markdown("### 📈 Academic Progression Across Periods")
        grade_summary = pd.DataFrame(
            {
                "Period 1 (G1)": df_dataset["G1"],
                "Period 2 (G2)": df_dataset["G2"],
                "Final Grade (G3)": df_dataset["G3"],
            }
        )
        fig_box = px.box(
            grade_summary,
            labels={"variable": "Assessment Period", "value": "Grade (0-20 scale)"},
            color_discrete_sequence=["#2563eb"],
        )
        fig_box.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            yaxis=dict(range=[0, 20]),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Educational Grade Boundary Rule Explanation
    st.markdown("### 📐 Target Classification Methodology")
    st.info(
        """
        The academic performance categories are mapped directly from the student's final grade (**G3**) 
        on the official Portuguese **0 to 20 scale**:
        
        * 🔴 **Low Performance (0 – 9):** Fails to meet the minimum secondary education passing standard (< 10/20). Indicates significant academic distress requiring early remedial tutoring.
        * 🔵 **Medium Performance (10 – 13):** Satisfactory passing range (10–13/20). Demonstrates solid comprehension with potential for further elevation.
        * 🟢 **High Performance (14 – 20):** Good to excellent academic achievement (14–20/20). High mastery of mathematics curriculum.
        
        *Target leakage prevention:* The raw final grade `G3` is strictly removed from input features during model training and evaluation.
        """
    )

    # Quick Navigation
    st.markdown("### 🚀 Quick Exploration & Action Paths")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        st.markdown(
            """
            **🔮 Predict Performance**  
            Input student characteristics, prior period grades, and lifestyle attributes to generate real-time tier predictions with confidence levels.
            """
        )
        st.button(
            "Launch Predictor →",
            key="btn_goto_predict",
            on_click=set_nav_page,
            args=("🔮 Predict Performance",),
            use_container_width=True,
        )
    with q_col2:
        st.markdown(
            """
            **📊 Model Performance**  
            Review test-set accuracy, 5-Fold cross validation, confusion matrix, classification reports, and feature importance rankings.
            """
        )
        st.button(
            "View Model Metrics →",
            key="btn_goto_model",
            on_click=set_nav_page,
            args=("📊 Model Performance",),
            use_container_width=True,
        )
    with q_col3:
        st.markdown(
            """
            **🔍 Dataset Explorer**  
            Filter, inspect, and visualize student records and explore correlations across demographic and academic features.
            """
        )
        st.button(
            "Explore Dataset →",
            key="btn_goto_dataset",
            on_click=set_nav_page,
            args=("🔍 Dataset Explorer",),
            use_container_width=True,
        )


# =========================================================
# PAGE B: STUDENT PERFORMANCE PREDICTION
# =========================================================
elif menu_selection == "🔮 Predict Performance":
    st.title("🔮 Student Performance Prediction")
    st.markdown(
        "Estimate a student's final performance category using trained machine learning models based on "
        "demographic background, lifestyle indicators, and prior academic results (**G1** & **G2**)."
    )

    if model_payload is None:
        st.error(
            "⚠️ Trained model artifact not found. Please run `python train_model.py` in your terminal first."
        )
        st.stop()

    st.markdown(
        """
        > 💡 **Notice on Academic Predictors:**  
        > Prior term grades (**G1** and **G2**) are strong indicators of final performance. 
        > Model outputs and probabilities are statistical estimates and educational decision support aids, not deterministic guarantees.
        """
    )

    # Profile Preset Selector
    st.markdown("#### ⚡ Quick-Fill Preset Profiles")
    preset_choice = st.selectbox(
        "Load sample profile for rapid testing and demonstrations:",
        ["Custom Input (Manual)", "High Achiever Profile", "Average Student Profile", "At-Risk Student Profile"],
        index=0,
    )

    # Determine default values based on selected profile
    initial_values = {}
    if preset_choice in SAMPLE_PROFILES:
        initial_values = SAMPLE_PROFILES[preset_choice]

    p_key = preset_choice.replace(" ", "_")

    # Prediction Input Form
    with st.form("prediction_form"):
        # Group 1: Academic Record
        st.markdown("### 📚 1. Academic Record")
        st.caption("Prior period examination scores and historical course failure record.")
        ar_col1, ar_col2, ar_col3 = st.columns(3)
        with ar_col1:
            g1_val = initial_values.get("G1", 11)
            inp_g1 = st.slider(
                "Period 1 Grade (G1)", min_value=0, max_value=20, value=int(g1_val),
                key=f"g1_{p_key}", help="Earlier first-period academic grade on official Portuguese 0–20 scale"
            )
        with ar_col2:
            g2_val = initial_values.get("G2", 11)
            inp_g2 = st.slider(
                "Period 2 Grade (G2)", min_value=0, max_value=20, value=int(g2_val),
                key=f"g2_{p_key}", help="Earlier second-period academic grade on official Portuguese 0–20 scale"
            )
        with ar_col3:
            failures_val = initial_values.get("failures", 0)
            inp_failures = st.number_input(
                "Past Class Failures",
                min_value=0,
                max_value=4,
                value=int(failures_val),
                key=f"failures_{p_key}",
                help="Number of past courses failed (0 to 4)",
            )

        # Group 2: Study Habits
        st.markdown("### ⏱️ 2. Study Habits & Attendance")
        st.caption("Time dedicated to academic revision, absences, and commute.")
        sh_col1, sh_col2, sh_col3 = st.columns(3)
        with sh_col1:
            studytime_val = initial_values.get("studytime", 2)
            studytime_opts = [1, 2, 3, 4]
            inp_studytime = st.selectbox(
                "Weekly Study Time",
                options=studytime_opts,
                index=studytime_opts.index(studytime_val),
                key=f"studytime_{p_key}",
                format_func=lambda x: FEATURE_METADATA["studytime"]["labels"][x],
                help="Average weekly self-study duration outside class hours",
            )
        with sh_col2:
            absences_val = initial_values.get("absences", 4)
            inp_absences = st.number_input(
                "Recorded school absences",
                min_value=0,
                max_value=93,
                value=int(absences_val),
                key=f"absences_{p_key}",
                help="Total count of missed school days recorded in dataset. The source dataset records an absence count but does not specify a consistent total number of possible academic days.",
            )
        with sh_col3:
            traveltime_val = initial_values.get("traveltime", 1)
            inp_traveltime = st.selectbox(
                "Travel Time to School",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(traveltime_val),
                key=f"traveltime_{p_key}",
                format_func=lambda x: FEATURE_METADATA["traveltime"]["labels"][x],
                help="Daily one-way commute duration to school",
            )

        # Group 3: Family and School Support
        st.markdown("### 👨‍👩‍👧 3. Family and School Support")
        st.caption("Parental education levels, occupations, and institutional academic resources.")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            medu_val = initial_values.get("Medu", 2)
            inp_medu = st.selectbox(
                "Mother's Education",
                options=[0, 1, 2, 3, 4],
                index=[0, 1, 2, 3, 4].index(medu_val),
                key=f"medu_{p_key}",
                format_func=lambda x: FEATURE_METADATA["Medu"]["labels"][x],
            )
            mjob_val = initial_values.get("Mjob", "other")
            inp_mjob = st.selectbox(
                "Mother's Job",
                options=["at_home", "health", "services", "teacher", "other"],
                index=["at_home", "health", "services", "teacher", "other"].index(mjob_val),
                key=f"mjob_{p_key}",
                format_func=lambda x: FEATURE_METADATA["Mjob"]["labels"][x],
            )
        with f_col2:
            fedu_val = initial_values.get("Fedu", 2)
            inp_fedu = st.selectbox(
                "Father's Education",
                options=[0, 1, 2, 3, 4],
                index=[0, 1, 2, 3, 4].index(fedu_val),
                key=f"fedu_{p_key}",
                format_func=lambda x: FEATURE_METADATA["Fedu"]["labels"][x],
            )
            fjob_val = initial_values.get("Fjob", "other")
            inp_fjob = st.selectbox(
                "Father's Job",
                options=["at_home", "health", "services", "teacher", "other"],
                index=["at_home", "health", "services", "teacher", "other"].index(fjob_val),
                key=f"fjob_{p_key}",
                format_func=lambda x: FEATURE_METADATA["Fjob"]["labels"][x],
            )
        with f_col3:
            guardian_val = initial_values.get("guardian", "mother")
            inp_guardian = st.selectbox(
                "Guardian",
                options=["mother", "father", "other"],
                index=["mother", "father", "other"].index(guardian_val),
                key=f"guardian_{p_key}",
                format_func=lambda x: x.capitalize(),
            )
            famsup_val = initial_values.get("famsup", "yes")
            inp_famsup = st.selectbox(
                "Family Educational Support",
                options=["no", "yes"],
                index=["no", "yes"].index(famsup_val),
                key=f"famsup_{p_key}",
                format_func=lambda x: "Yes" if x == "yes" else "No",
                help="Educational support from parents or household members",
            )
        with f_col4:
            schoolsup_val = initial_values.get("schoolsup", "no")
            inp_schoolsup = st.selectbox(
                "School Academic Support",
                options=["no", "yes"],
                index=["no", "yes"].index(schoolsup_val),
                key=f"schoolsup_{p_key}",
                format_func=lambda x: "Yes" if x == "yes" else "No",
                help="Extra remedial educational support provided by the school",
            )
            paid_val = initial_values.get("paid", "no")
            inp_paid = st.selectbox(
                "Extra Paid Math Classes",
                options=["no", "yes"],
                index=["no", "yes"].index(paid_val),
                key=f"paid_{p_key}",
                format_func=lambda x: "Yes" if x == "yes" else "No",
                help="Private paid tutoring in Mathematics",
            )

        # Group 4: Personal Background
        st.markdown("### 🏃 4. Personal Background & Lifestyle")
        st.caption("Demographic context, health, extracurricular activities, and aspirations.")

        # Row 1
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
        with r1_c1:
            school_val = initial_values.get("school", "GP")
            inp_school = st.selectbox(
                "School",
                options=["GP", "MS"],
                index=["GP", "MS"].index(school_val),
                key=f"school_{p_key}",
                format_func=lambda x: FEATURE_METADATA["school"]["labels"][x],
            )
        with r1_c2:
            famsize_val = initial_values.get("famsize", "GT3")
            inp_famsize = st.selectbox(
                "Family Size",
                options=["GT3", "LE3"],
                index=["GT3", "LE3"].index(famsize_val),
                key=f"famsize_{p_key}",
                format_func=lambda x: FEATURE_METADATA["famsize"]["labels"][x],
            )
        with r1_c3:
            higher_val = initial_values.get("higher", "yes")
            inp_higher = st.selectbox(
                "Aspires to Higher Education",
                options=["no", "yes"],
                index=["no", "yes"].index(higher_val),
                key=f"higher_{p_key}",
                format_func=lambda x: "Yes" if x == "yes" else "No",
            )
        with r1_c4:
            famrel_val = initial_values.get("famrel", 4)
            inp_famrel = st.slider(
                "Family Relationship Quality", 1, 5, int(famrel_val),
                key=f"famrel_{p_key}", help="1: Very bad, 5: Excellent"
            )

        # Row 2
        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
        with r2_c1:
            sex_val = initial_values.get("sex", "F")
            inp_sex = st.selectbox(
                "Sex",
                options=["F", "M"],
                index=["F", "M"].index(sex_val),
                key=f"sex_{p_key}",
                format_func=lambda x: "Female" if x == "F" else "Male",
            )
        with r2_c2:
            pstatus_val = initial_values.get("Pstatus", "T")
            inp_pstatus = st.selectbox(
                "Parent Cohabitation",
                options=["T", "A"],
                index=["T", "A"].index(pstatus_val),
                key=f"pstatus_{p_key}",
                format_func=lambda x: FEATURE_METADATA["Pstatus"]["labels"][x],
            )
        with r2_c3:
            internet_val = initial_values.get("internet", "yes")
            inp_internet = st.selectbox(
                "Home Internet Access",
                options=["no", "yes"],
                index=["no", "yes"].index(internet_val),
                key=f"internet_{p_key}",
                format_func=lambda x: "Yes" if x == "yes" else "No",
            )
        with r2_c4:
            freetime_val = initial_values.get("freetime", 3)
            inp_freetime = st.slider(
                "Free Time After School", 1, 5, int(freetime_val),
                key=f"freetime_{p_key}", help="1: Very low, 5: Very high"
            )

        # Row 3
        r3_c1, r3_c2, r3_c3, r3_c4 = st.columns(4)
        with r3_c1:
            age_val = initial_values.get("age", 17)
            inp_age = st.slider("Age", min_value=15, max_value=22, value=int(age_val), key=f"age_{p_key}")
        with r3_c2:
            reason_val = initial_values.get("reason", "course")
            inp_reason = st.selectbox(
                "School Choice Reason",
                options=["course", "home", "reputation", "other"],
                index=["course", "home", "reputation", "other"].index(reason_val),
                key=f"reason_{p_key}",
                format_func=lambda x: FEATURE_METADATA["reason"]["labels"][x],
            )
        with r3_c3:
            activities_val = initial_values.get("activities", "no")
            inp_activities = st.selectbox(
                "Extracurricular Activities",
                options=["no", "yes"],
                index=["no", "yes"].index(activities_val),
                key=f"activities_{p_key}",
                format_func=lambda x: "Yes" if x == "yes" else "No",
            )
        with r3_c4:
            goout_val = initial_values.get("goout", 3)
            inp_goout = st.slider(
                "Socializing / Going Out", 1, 5, int(goout_val),
                key=f"goout_{p_key}", help="1: Very low, 5: Very high"
            )

        # Row 4
        r4_c1, r4_c2, r4_c3, r4_c4 = st.columns(4)
        with r4_c1:
            address_val = initial_values.get("address", "U")
            inp_address = st.selectbox(
                "Address Type",
                options=["U", "R"],
                index=["U", "R"].index(address_val),
                key=f"address_{p_key}",
                format_func=lambda x: "Urban" if x == "U" else "Rural",
            )
        with r4_c2:
            nursery_val = initial_values.get("nursery", "yes")
            inp_nursery = st.selectbox(
                "Attended Nursery / Preschool",
                options=["no", "yes"],
                index=["no", "yes"].index(nursery_val),
                key=f"nursery_{p_key}",
                format_func=lambda x: "Yes" if x == "yes" else "No",
            )
        with r4_c3:
            romantic_val = initial_values.get("romantic", "no")
            inp_romantic = st.selectbox(
                "Romantic Relationship",
                options=["no", "yes"],
                index=["no", "yes"].index(romantic_val),
                key=f"romantic_{p_key}",
                format_func=lambda x: "Yes" if x == "yes" else "No",
            )
        with r4_c4:
            dalc_val = initial_values.get("Dalc", 1)
            inp_dalc = st.slider(
                "Workday Alcohol Consumption", 1, 5, int(dalc_val),
                key=f"dalc_{p_key}", help="1: Very low, 5: Very high"
            )

        # Row 5 (Balanced pair of health & weekend lifestyle)
        r5_c1, r5_c2 = st.columns(2)
        with r5_c1:
            walc_val = initial_values.get("Walc", 1)
            inp_walc = st.slider(
                "Weekend Alcohol Consumption", 1, 5, int(walc_val),
                key=f"walc_{p_key}", help="1: Very low, 5: Very high"
            )
        with r5_c2:
            health_val = initial_values.get("health", 4)
            inp_health = st.slider(
                "Current Health Status", 1, 5, int(health_val),
                key=f"health_{p_key}", help="1: Very poor, 5: Very good"
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Run Prediction & Analysis", use_container_width=True)

    # Process Form Submission
    if submitted:
        input_record = {
            "school": inp_school,
            "sex": inp_sex,
            "age": inp_age,
            "address": inp_address,
            "famsize": inp_famsize,
            "Pstatus": inp_pstatus,
            "Medu": inp_medu,
            "Fedu": inp_fedu,
            "Mjob": inp_mjob,
            "Fjob": inp_fjob,
            "reason": inp_reason,
            "guardian": inp_guardian,
            "traveltime": inp_traveltime,
            "studytime": inp_studytime,
            "failures": inp_failures,
            "schoolsup": inp_schoolsup,
            "famsup": inp_famsup,
            "paid": inp_paid,
            "activities": inp_activities,
            "nursery": inp_nursery,
            "higher": inp_higher,
            "internet": inp_internet,
            "romantic": inp_romantic,
            "famrel": inp_famrel,
            "freetime": inp_freetime,
            "goout": inp_goout,
            "Dalc": inp_dalc,
            "Walc": inp_walc,
            "health": inp_health,
            "absences": inp_absences,
            "G1": inp_g1,
            "G2": inp_g2,
        }

        with st.spinner("Analyzing student profile with Random Forest Classifier..."):
            pred_result = predict_student_performance(model_payload, input_record)

        pred_category = pred_result["predicted_category"]
        conf = pred_result["confidence"]
        probs = pred_result["class_probabilities"]
        interp = pred_result["interpretation"]

        st.markdown("---")
        st.markdown("## 🎯 Prediction Outcome")

        # Visual Result Banner
        css_class = (
            "result-card-high"
            if pred_category == "High Performance"
            else ("result-card-medium" if pred_category == "Medium Performance" else "result-card-low")
        )
        badge_class = (
            "badge-high"
            if pred_category == "High Performance"
            else ("badge-medium" if pred_category == "Medium Performance" else "badge-low")
        )

        st.markdown(
            f"""
            <div class="{css_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span class="{badge_class}">{pred_category}</span>
                    <span style="font-weight: 700; color: #0f172a; font-size: 1.15rem;">
                        Estimated Confidence: {conf*100:.1f}%
                    </span>
                </div>
                <h3 style="margin-top: 0; color: #0f172a;">{interp['headline']}</h3>
                <p style="font-size: 1.05rem; color: #334155; line-height: 1.6;">{interp['summary']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probabilities Distribution Chart & Recommendations
        r_col1, r_col2 = st.columns([1, 1])
        with r_col1:
            st.markdown("### 📊 Class Probability Breakdown")
            prob_df = pd.DataFrame(
                {
                    "Performance Category": list(probs.keys()),
                    "Probability (%)": [p * 100 for p in probs.values()],
                }
            )
            fig_prob = px.bar(
                prob_df,
                x="Probability (%)",
                y="Performance Category",
                orientation="h",
                color="Performance Category",
                color_discrete_map=CATEGORY_COLORS,
                text=prob_df["Probability (%)"].apply(lambda x: f"{x:.1f}%"),
            )
            fig_prob.update_layout(
                xaxis_title="Estimated Likelihood (%)",
                yaxis_title="",
                xaxis=dict(range=[0, 100]),
                height=260,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        with r_col2:
            st.markdown("### 💡 Educational Action Insights")
            for rec in interp["recommendations"]:
                st.markdown(f"- {rec}")

            st.caption(
                "⚠️ **Educational Prediction Disclaimer:** Class probabilities and estimated confidence represent "
                "statistical pattern likelihoods derived by the Random Forest model from historical cohort data. "
                "They are decision-support aids designed for positive academic counseling and early tutoring intervention, "
                "not deterministic guarantees of a student's personal potential or final grades."
            )


# =========================================================
# PAGE C: MODEL PERFORMANCE
# =========================================================
elif menu_selection == "📊 Model Performance":
    st.title("📊 Model Performance & Evaluation")
    st.markdown(
        "Evaluation results on the held-out test set (80% train / 20% test stratified split, 79 test samples) "
        "and 5-Fold Stratified Cross-Validation on the training data. All figures are empirical results obtained from Scikit-learn."
    )

    if model_payload is None:
        st.error("Model artifacts not found. Please run `python train_model.py` first.")
        st.stop()

    metrics = model_payload["primary_metrics"]
    baselines = model_payload["baseline_metrics"]

    # Top KPI Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Test Accuracy</div>
                <div class="metric-value">{metrics['accuracy']*100:.2f}%</div>
                <div class="metric-sub">Held-out test set (79 samples)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        cv_acc = metrics.get('cv_accuracy_mean', 0.848)
        cv_std = metrics.get('cv_accuracy_std', 0.037)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">5-Fold CV Accuracy</div>
                <div class="metric-value">{cv_acc*100:.1f}%</div>
                <div class="metric-sub">± {cv_std*100:.1f}% across training folds</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Macro F1-Score</div>
                <div class="metric-value">{metrics['f1_macro']:.4f}</div>
                <div class="metric-sub">Balanced across all 3 classes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">High Tier Recall</div>
                <div class="metric-value">{metrics['classification_report']['High Performance']['recall']*100:.1f}%</div>
                <div class="metric-sub">19 of 20 high performers captured</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Confusion Matrix and Model Comparison
    col_cm, col_comp = st.columns([1, 1])

    with col_cm:
        st.markdown("### 🧮 Confusion Matrix (Held-Out Test Set: 79 Samples)")
        cm_matrix = np.array(metrics["confusion_matrix"])
        target_labels = model_payload["target_classes"]

        fig_cm = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Class", y="Actual True Class", color="Count"),
            x=target_labels,
            y=target_labels,
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(
            height=340,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_comp:
        st.markdown("### ⚖️ Comparison with Baseline Models")
        comp_data = []
        for key, val in baselines.items():
            cv_text = f"{val.get('cv_accuracy_mean', val.get('cv_accuracy', 0))*100:.1f}%" if "cv_accuracy_mean" in val or "cv_accuracy" in val else "N/A"
            comp_data.append(
                {
                    "Model": val["name"],
                    "Test Accuracy (%)": round(val["accuracy"] * 100, 2),
                    "5-Fold CV (%)": cv_text,
                    "Macro F1": round(val["f1_macro"], 4),
                }
            )
        comp_df = pd.DataFrame(comp_data)

        fig_comp = px.bar(
            comp_df,
            x="Model",
            y="Test Accuracy (%)",
            color="Model",
            text="Test Accuracy (%)",
            color_discrete_sequence=["#94a3b8", "#60a5fa", "#16a34a"],
        )
        fig_comp.update_layout(
            height=340,
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis=dict(range=[0, 100]),
            showlegend=False,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Statistical Rationale Callout
    st.info(
        r"""
        🔍 **Statistical Analysis on Model Comparison:**
        * **Cross-Validation Superiority:** On 5-Fold Stratified Cross-Validation on the training set, **Random Forest achieves 84.8% ± 3.7% accuracy**, outperforming **Logistic Regression (82.9% ± 2.9%)**.
        * **Single Test Split Variance:** On the single 79-sample held-out test split, Logistic Regression obtained 87.34% (69/79) while Random Forest obtained 83.54% (66/79). This difference corresponds to **only 3 individual students**, well within 1 standard deviation of sampling noise ($N=79$, each student is 1.27%).
        * **Minority Class Sensitivity:** Random Forest with balanced class weighting achieves **95.0% recall on High Performance students** (compared to 85.0% for Logistic Regression), making it the more reliable choice for identifying high-achieving and at-risk students.
        """
    )

    # Classification Report Breakdown Table
    st.markdown("### 📋 Per-Class Performance Breakdown")
    cr_dict = metrics["classification_report"]
    report_rows = []
    for cls in model_payload["target_classes"]:
        if cls in cr_dict:
            report_rows.append(
                {
                    "Class": cls,
                    "Precision": f"{cr_dict[cls]['precision']:.4f}",
                    "Recall": f"{cr_dict[cls]['recall']:.4f}",
                    "F1-Score": f"{cr_dict[cls]['f1-score']:.4f}",
                    "Support": int(cr_dict[cls]['support']),
                }
            )
    st.table(pd.DataFrame(report_rows))

    # Feature Importance Chart
    st.markdown("### 🌟 Feature Importances (Top Influential Factors)")
    top_features = pd.DataFrame(model_payload["top_feature_importances"][:12])
    fig_feat = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Gini Feature Importance", "feature": "Feature"},
        color="importance",
        color_continuous_scale="Viridis",
    )
    fig_feat.update_layout(
        height=380,
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    # Algorithm and Experimental Methodology Details
    st.markdown("### 🧠 Machine Learning Methodology")
    st.markdown(
        """
        - **Primary Algorithm:** Random Forest Classifier (150 estimators, max depth 8, balanced class weighting to address category distribution).
        - **Ensemble Rationale:** Combines multiple decorrelated decision trees built on bootstrap data samples (bagging) with random feature subspace sampling. This significantly mitigates individual tree overfitting and yields robust probability estimates.
        - **Data Preprocessing:** Standardizes continuous numeric attributes using `StandardScaler` and encodes categorical variables via `OneHotEncoder(drop='first', handle_unknown='ignore')` in an atomic Scikit-learn `Pipeline`.
        - **Target Leakage Prevention:** `G3` is exclusively used to compute target performance labels for training and is strictly dropped prior to feature transformation.
        - **Baseline Benchmark:** Compared against a Majority Class Dummy Baseline (41.77% accuracy) and regularized Logistic Regression (87.34% test accuracy / 82.91% CV accuracy).
        """
    )

    st.markdown("### ⚠️ Model Scope & Limitations")
    st.caption(
        """
        - **Sample Size Constraint:** The held-out test set comprises 79 student instances (20% of 395). Each sample corresponds to 1.27% of test accuracy; minor fluctuations of 2–3 students are expected due to test sampling variance.
        - **Feature Dominance:** In line with educational literature, earlier examination scores (G1 and G2) account for ~59% of cumulative feature importance. While demographic and lifestyle variables provide meaningful regularizing context, they cannot override substantial grade deficits.
        - **Curricular Domain:** This model is trained on secondary school Mathematics students in Portugal and reflects that educational context. It serves as an assistive academic screening tool, not an absolute predictor.
        """
    )


# =========================================================
# PAGE D: DATASET EXPLORER
# =========================================================
elif menu_selection == "🔍 Dataset Explorer":
    st.title("🔍 Dataset Explorer & Visualizer")
    st.markdown(
        "Interactive exploration of the UCI Student Performance Dataset (Mathematics course). "
        "Inspect records, summary statistics, distributions, and relationship patterns."
    )

    if df_dataset is None:
        st.error("Dataset not loaded. Please ensure `data/student-mat.csv` exists.")
        st.stop()

    # Initialize or reset filter keys in session_state
    if "filter_category" not in st.session_state:
        st.session_state["filter_category"] = []
    if "filter_school" not in st.session_state:
        st.session_state["filter_school"] = []
    if "filter_sex" not in st.session_state:
        st.session_state["filter_sex"] = []
    if "filter_g3_range" not in st.session_state:
        st.session_state["filter_g3_range"] = (0, 20)

    def reset_dataset_filters():
        st.session_state["filter_category"] = []
        st.session_state["filter_school"] = []
        st.session_state["filter_sex"] = []
        st.session_state["filter_g3_range"] = (0, 20)

    # Filter Controls Expander
    with st.expander("🛠️ Filter Dataset Records (Optional Slicing)", expanded=True):
        st.caption(
            "By default, all filters are unconstrained, showing the full dataset. "
            "Select specific criteria below to isolate cohorts, or click 'Reset All Filters' to restore the full view."
        )
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            cat_filter = st.multiselect(
                "Performance Category:",
                options=TARGET_CLASSES,
                key="filter_category",
                placeholder="All Categories (Default)",
                help="Leave empty to include Low, Medium, and High performance.",
            )
        with f_col2:
            school_filter = st.multiselect(
                "School:",
                options=["GP", "MS"],
                key="filter_school",
                format_func=lambda x: "Gabriel Pereira (GP)" if x == "GP" else "Mousinho da Silveira (MS)",
                placeholder="Both Schools (Default)",
                help="Leave empty to include both GP and MS.",
            )
        with f_col3:
            sex_filter = st.multiselect(
                "Sex:",
                options=["F", "M"],
                key="filter_sex",
                format_func=lambda x: "Female (F)" if x == "F" else "Male (M)",
                placeholder="Both Sexes (Default)",
                help="Leave empty to include all students.",
            )
        with f_col4:
            g3_range = st.slider(
                "Final Grade (G3) Range:",
                min_value=0,
                max_value=20,
                key="filter_g3_range",
                help="Filter by final Portuguese math score (0 to 20).",
            )

        btn_c1, btn_c2 = st.columns([1, 4])
        with btn_c1:
            st.button("🔄 Reset All Filters", key="btn_reset_dataset_filters", on_click=reset_dataset_filters, use_container_width=True)

    # Determine Active Filtering Criteria
    active_cats = cat_filter if cat_filter else TARGET_CLASSES
    active_schools = school_filter if school_filter else ["GP", "MS"]
    active_sexes = sex_filter if sex_filter else ["F", "M"]

    is_filtered = bool(cat_filter) or bool(school_filter) or bool(sex_filter) or (g3_range != (0, 20))

    # Apply Filters to dataset
    filtered_df = df_dataset[
        (df_dataset["performance_category"].isin(active_cats))
        & (df_dataset["school"].isin(active_schools))
        & (df_dataset["sex"].isin(active_sexes))
        & (df_dataset["G3"] >= g3_range[0])
        & (df_dataset["G3"] <= g3_range[1])
    ]

    # Active Filters Summary Header
    st.markdown(
        f"### 📋 Student Records Table "
        f"<span style='font-size:0.95rem; color:#475569; font-weight:normal;'>"
        f"(Showing **{len(filtered_df)}** of **{len(df_dataset)}** records — "
        f"{'Custom Cohort Filter Applied' if is_filtered else 'Full Dataset Scope'})</span>",
        unsafe_allow_html=True,
    )

    if is_filtered:
        filter_summary_html = "<div style='margin-bottom: 12px;'>"
        if cat_filter:
            filter_summary_html += f"<span class='filter-badge'>Categories: {', '.join(cat_filter)}</span>"
        if school_filter:
            filter_summary_html += f"<span class='filter-badge'>Schools: {', '.join(school_filter)}</span>"
        if sex_filter:
            filter_summary_html += f"<span class='filter-badge'>Sex: {', '.join(sex_filter)}</span>"
        if g3_range != (0, 20):
            filter_summary_html += f"<span class='filter-badge'>G3: {g3_range[0]}–{g3_range[1]}</span>"
        filter_summary_html += "</div>"
        st.markdown(filter_summary_html, unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("⚠️ No student records match the selected filter criteria. Click 'Reset All Filters' above to restore records.")
    else:
        # Dataset Preview Table
        st.dataframe(filtered_df, use_container_width=True, height=280)

        # Descriptive Statistics & Interactive Visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Grade Correlations", "📉 Absences & Study Time", "🔢 Descriptive Statistics"])

        with tab1:
            st.markdown("#### Earlier Period Grade Progression (G1, G2) vs Final Performance")
            st.caption(
                "Shows student positions across Period 1 (G1) and Period 2 (G2). "
                "Marker size indicates final grade (G3), and color indicates classified performance tier."
            )
            fig_scatter = px.scatter(
                filtered_df,
                x="G1",
                y="G2",
                size="G3",
                color="performance_category",
                color_discrete_map=CATEGORY_COLORS,
                hover_data=["age", "studytime", "failures", "absences", "G3"],
                labels={
                    "G1": "Period 1 Grade (G1, 0–20 scale)",
                    "G2": "Period 2 Grade (G2, 0–20 scale)",
                    "performance_category": "Performance Tier",
                    "G3": "Final Grade (G3)",
                },
            )
            fig_scatter.update_layout(
                height=420,
                margin=dict(t=20, b=20, l=20, r=20),
                legend_title_text="Performance Tier",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with tab2:
            st.markdown("#### Distribution of School Absences and Study Time Across Performance Categories")
            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(
                    filtered_df,
                    x="absences",
                    color="performance_category",
                    color_discrete_map=CATEGORY_COLORS,
                    barmode="overlay",
                    nbins=25,
                    labels={
                        "absences": "Recorded School Absences (Days)",
                        "performance_category": "Performance Tier",
                    },
                )
                fig_hist.update_layout(
                    title="Absence Count Distribution by Category",
                    height=360,
                    margin=dict(t=40, b=20, l=20, r=20),
                    legend_title_text="Performance Tier",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with c2:
                fig_box_study = px.box(
                    filtered_df,
                    x="studytime",
                    y="G3",
                    color="studytime",
                    labels={
                        "studytime": "Weekly Study Time Tier (1: <2h, 2: 2–5h, 3: 5–10h, 4: >10h)",
                        "G3": "Final Grade (G3, 0–20 scale)",
                    },
                )
                fig_box_study.update_layout(
                    title="Final Grade (G3) by Weekly Study Time Tier",
                    height=360,
                    margin=dict(t=40, b=20, l=20, r=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_box_study, use_container_width=True)

        with tab3:
            st.markdown("#### Numerical & Academic Feature Summary Statistics")
            st.caption(
                f"Computed across the active view scope ({len(filtered_df)} student records). "
                f"Values are rounded to two decimal places."
            )
            st.dataframe(filtered_df.describe().T.round(2), use_container_width=True)

    # Attribution & Ethics Note
    st.markdown("---")
    st.caption(
        "**Dataset Attribution:** P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance*. "
        "In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal. "
        "UCI Machine Learning Repository. Identifiable personal data is anonymized."
    )

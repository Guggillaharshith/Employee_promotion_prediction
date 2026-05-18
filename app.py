# ============================================================
# EMPLOYEE PROMOTION PREDICTION SYSTEM
# Streamlit Deployment Project
# ============================================================

# Save as:
# app.py

# Run:
# streamlit run app.py

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Employee_promotion_prediction",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #141E30, #243B55);
}

.main-title {
    text-align: center;
    color: white;
    font-size: 45px;
    font-weight: bold;
    margin-bottom: 20px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}

.result-success {
    background-color: #28a745;
    color: white;
    padding: 20px;
    border-radius: 15px;
    font-size: 25px;
    font-weight: bold;
    text-align: center;
}

.result-fail {
    background-color: #dc3545;
    color: white;
    padding: 20px;
    border-radius: 15px;
    font-size: 25px;
    font-weight: bold;
    text-align: center;
}

.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown("""
<div class="main-title">
📈 Employee Promotion Prediction
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("employee_promotion_dataset.csv")

# ============================================================
# DATA PREPROCESSING
# ============================================================

# Fill missing values
from pandas.api.types import is_numeric_dtype

for col in df.columns:
    if is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mean())
    else:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val[0])
        else:
            df[col] = df[col].fillna('Unknown')

# Encode categorical columns
encoder = LabelEncoder()

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# ============================================================
# FEATURES & TARGET
# ============================================================

X = df.drop("is_promoted", axis=1)

y = df["is_promoted"]

# ============================================================
# SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# TRAIN MODELS
# ============================================================

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# ============================================================
# PREDICTIONS
# ============================================================

dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# ============================================================
# ACCURACY
# ============================================================

dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

# ============================================================
# SIDEBAR MENU
# ============================================================

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dataset",
        "Model Accuracy",
        "Prediction",
        "Visualization"
    ]
)

# ============================================================
# DATASET SECTION
# ============================================================

if menu == "Dataset":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Employee Promotion Dataset")

    st.dataframe(df)

    st.write("Dataset Shape:", df.shape)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# MODEL ACCURACY SECTION
# ============================================================

elif menu == "Model Accuracy":

    st.subheader("📌 Accuracy Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
        <h3>Decision Tree</h3>
        <h1>{dt_acc:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
        <h3>Random Forest</h3>
        <h1>{rf_acc:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
        <h3>XGBoost</h3>
        <h1>{xgb_acc:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

    # Accuracy Visualization
    st.subheader("📈 Accuracy Visualization")

    models = [
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ]

    accuracies = [
        dt_acc,
        rf_acc,
        xgb_acc
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(models, accuracies)

    ax.set_ylabel("Accuracy")

    ax.set_title("Model Comparison")

    st.pyplot(fig)

# ============================================================
# PREDICTION SECTION
# ============================================================

elif menu == "Prediction":

    st.subheader("🔮 Employee Promotion Prediction")

    # User Inputs
    department = st.selectbox(
        "Department",
        sorted(df['department'].unique())
    )

    region = st.selectbox(
        "Region",
        sorted(df['region'].unique())
    )

    education = st.selectbox(
        "Education",
        sorted(df['education'].unique())
    )

    gender = st.selectbox(
        "Gender",
        sorted(df['gender'].unique())
    )

    recruitment_channel = st.selectbox(
        "Recruitment Channel",
        sorted(df['recruitment_channel'].unique())
    )

    no_of_trainings = st.slider(
        "No of Trainings",
        1, 10, 2
    )

    age = st.slider(
        "Age",
        20, 60, 30
    )

    previous_year_rating = st.slider(
        "Previous Year Rating",
        1, 5, 3
    )

    length_of_service = st.slider(
        "Length of Service",
        1, 40, 5
    )

    KPIs_met = st.selectbox(
        "KPIs Met >80%",
        [0, 1]
    )

    awards_won = st.selectbox(
        "Awards Won",
        [0, 1]
    )

    avg_training_score = st.slider(
        "Average Training Score",
        40, 100, 70
    )

    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    if st.button("Predict Promotion"):

        input_data = pd.DataFrame({
            'department': [department],
            'region': [region],
            'education': [education],
            'gender': [gender],
            'recruitment_channel': [recruitment_channel],
            'no_of_trainings': [no_of_trainings],
            'age': [age],
            'previous_year_rating': [previous_year_rating],
            'length_of_service': [length_of_service],
            'KPIs_met >80%': [KPIs_met],
            'awards_won?': [awards_won],
            'avg_training_score': [avg_training_score]
        })

        # Encode categorical inputs
        for col in input_data.columns:

            if input_data[col].dtype == 'object':

                input_data[col] = encoder.fit_transform(
                    input_data[col]
                )

        # Random Forest Prediction
        prediction = rf_model.predict(input_data)[0]

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        if prediction == 1:

            st.markdown("""
            <div class="result-success">
            ✅ Employee WILL be Promoted
            </div>
            """, unsafe_allow_html=True)

            st.success("Reasons for Promotion:")

            reasons = [
                "High Training Score",
                "Excellent KPI Performance",
                "Good Previous Rating",
                "Strong Experience",
                "Consistent Performance"
            ]

            for reason in reasons:
                st.write("✔", reason)

            st.balloons()

        else:

            st.markdown("""
            <div class="result-fail">
            ❌ Employee will NOT be Promoted
            </div>
            """, unsafe_allow_html=True)

            st.error("Reasons for Not Promotion:")

            reasons = [
                "Low Training Score",
                "Poor KPI Performance",
                "Low Experience",
                "No Awards Achieved",
                "Weak Previous Rating"
            ]

            for reason in reasons:
                st.write("❌", reason)

# ============================================================
# VISUALIZATION SECTION
# ============================================================

elif menu == "Visualization":

    st.subheader("📊 Promotion Distribution")

    fig1, ax1 = plt.subplots(figsize=(6, 4))

    df['is_promoted'].value_counts().plot(
        kind='bar',
        ax=ax1
    )

    st.pyplot(fig1)

    st.subheader("📈 Feature Importance (Random Forest)")

    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by='Importance',
        ascending=False
    )

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.barh(
        importance_df['Feature'],
        importance_df['Importance']
    )

    st.pyplot(fig2)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<center>
<h4 style='color:white;'>
🚀 Employee Promotion Prediction using ML
</h4>
</center>
""", unsafe_allow_html=True)

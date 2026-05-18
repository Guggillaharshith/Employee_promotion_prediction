# ============================================================
# EMPLOYEE PROMOTION PREDICTION SYSTEM
# Fixed Streamlit Deployment Code
# ============================================================

# Save as: app.py

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Promotion Predictor",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #1f4e79;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: gray;
}

.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
}

.result-success {
    background-color: #d4edda;
    color: #155724;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}

.result-fail {
    background-color: #f8d7da;
    color: #721c24;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<p class="title">📈 Employee Promotion Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning using Decision Tree Classifier</p>',
    unsafe_allow_html=True
)

# ============================================================
# DATASET
# ============================================================

data = {
    'Experience': [2, 5, 1, 7, 3, 10, 4, 6, 8, 2],
    'Attendance': [80, 95, 60, 98, 75, 99, 85, 90, 97, 70],
    'PerformanceScore': [60, 88, 45, 92, 70, 96, 75, 85, 90, 55],
    'OvertimeHours': [5, 10, 2, 12, 4, 15, 6, 8, 11, 3],
    'Promoted': [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# ============================================================
# SHOW DATASET
# ============================================================

st.subheader("📊 Employee Dataset")

st.dataframe(df, use_container_width=True)

# ============================================================
# FEATURES & TARGET
# ============================================================

X = df[['Experience',
        'Attendance',
        'PerformanceScore',
        'OvertimeHours']]

y = df['Promoted']

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# MODEL TRAINING
# ============================================================

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

train_accuracy = model.score(X_train, y_train)

test_accuracy = model.score(X_test, y_test)

# ============================================================
# METRICS DISPLAY
# ============================================================

st.subheader("📌 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <h3>🎯 Accuracy</h3>
        <h1>{accuracy:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <h3>📚 Training Accuracy</h3>
        <h1>{train_accuracy:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <h3>🧪 Testing Accuracy</h3>
        <h1>{test_accuracy:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.subheader("📄 Classification Report")

report = classification_report(y_test, y_pred)

st.code(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("🔍 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    columns=['Predicted 0', 'Predicted 1'],
    index=['Actual 0', 'Actual 1']
)

st.dataframe(cm_df, use_container_width=True)

# ============================================================
# DECISION TREE VISUALIZATION
# ============================================================

st.subheader("🌳 Decision Tree")

fig, ax = plt.subplots(figsize=(14, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Not Promoted', 'Promoted'],
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)

st.pyplot(fig)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧑 Employee Details")

experience = st.sidebar.slider(
    "Experience (Years)",
    0, 15, 5
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    50, 100, 80
)

performance = st.sidebar.slider(
    "Performance Score",
    40, 100, 75
)

overtime = st.sidebar.slider(
    "Overtime Hours",
    0, 20, 5
)

# ============================================================
# PREDICT BUTTON
# ============================================================

predict = st.sidebar.button("🔮 Predict")

# ============================================================
# RESULT
# ============================================================

if predict:

    input_data = pd.DataFrame({
        'Experience': [experience],
        'Attendance': [attendance],
        'PerformanceScore': [performance],
        'OvertimeHours': [overtime]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.markdown(
            '<div class="result-success">Employee is likely to be promoted ✅</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-fail">Employee is unlikely to be promoted ❌</div>',
            unsafe_allow_html=True
        )

    st.markdown("### 📝 Prediction details")
    st.write(input_data)
    st.write(f"**Promotion probability:** {probability:.0%}")


    st.subheader("📢 Prediction Result")

    if prediction == 1:

        st.markdown("""
        <div class="result-success">
            ✅ Employee is Likely to be PROMOTED
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-fail">
            ❌ Employee is NOT Likely to be Promoted
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<center>
<h4>🚀 Streamlit ML Deployment Project</h4>
<p>Employee Promotion Prediction using Decision Tree</p>
</center>
""", unsafe_allow_html=True)
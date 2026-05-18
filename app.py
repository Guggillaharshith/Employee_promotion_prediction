# ============================================================
# EMPLOYEE PROMOTION PREDICTION SYSTEM
# Beautiful Streamlit UI
# ============================================================

# Save as: app.py

# Run:
# streamlit run app.py

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Employee Promotion Predictor",
    page_icon="📈",
    layout="centered"
)

# ============================================================
# CUSTOM CSS DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #1e3c72, #2a5298);
}

.main-title {
    text-align: center;
    color: white;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    color: #dbe9ff;
    font-size: 18px;
    margin-bottom: 40px;
}

.input-box {
    background-color: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}

.result-success {
    background-color: #28a745;
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

.result-fail {
    background-color: #dc3545;
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: white;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📈 Employee Promotion Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI-based Employee Promotion Prediction System</div>',
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
# MODEL TRAINING
# ============================================================

X = df[['Experience',
        'Attendance',
        'PerformanceScore',
        'OvertimeHours']]

y = df['Promoted']

model = DecisionTreeClassifier(random_state=42)

model.fit(X, y)

# ============================================================
# INPUT FORM
# ============================================================

st.markdown('<div class="input-box">', unsafe_allow_html=True)

st.subheader("🧑 Employee Details")

experience = st.slider(
    "Experience (Years)",
    min_value=0,
    max_value=15,
    value=5
)

attendance = st.slider(
    "Attendance Percentage",
    min_value=50,
    max_value=100,
    value=80
)

performance = st.slider(
    "Performance Score",
    min_value=40,
    max_value=100,
    value=75
)

overtime = st.slider(
    "Overtime Hours",
    min_value=0,
    max_value=20,
    value=5
)

predict_button = st.button("🔮 Predict Promotion")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_data = pd.DataFrame({
        'Experience': [experience],
        'Attendance': [attendance],
        'PerformanceScore': [performance],
        'OvertimeHours': [overtime]
    })

    prediction = model.predict(input_data)[0]

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:

        st.markdown("""
        <div class="result-success">
            ✅ Employee is Likely to be PROMOTED
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

    else:

        st.markdown("""
        <div class="result-fail">
            ❌ Employee is NOT Likely to be Promoted
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <h4>🚀 Streamlit Machine Learning Project</h4>
    <p>Employee Promotion Prediction using Decision Tree</p>
</div>
""", unsafe_allow_html=True)
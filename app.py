# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------- Streamlit Page Config -------------------------
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------- Custom CSS for styling -------------------------
st.markdown("""
<style>
/* Remove default Streamlit padding and max width */
.block-container {
    padding: 0rem 2rem 1rem 2rem;
    max-width: 1200px;
    margin: auto;
}

/* Reduce column gap */
.css-1lcbmhc.e1fqkh3o3 {
    gap: 1rem;
}

/* Card-style sections */
.section-card {
    background-color: white;
    padding: 20px 25px;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 10px;
}

/* Header & footer full width and fixed padding */
.header, .footer {
    margin: 0 -2rem;
    padding: 40px 20px 30px 20px;  /* increased top padding */
    text-align: center;
}
.header {
    background-color: #4B8BBE;
    border-radius: 0px; /* remove border radius so full width */
}
.header h1, .header p {
    color: white;
    margin: 0;
    padding: 0;
}
.footer {
    color: #888;
}

/* Buttons styling */
.stButton>button {
    background-color: #4B8BBE;
    color: white;
    font-size: 16px;
    font-weight: bold;
    height: 50px;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #3573a1;
}

/* Reduce markdown header spacing */
h1, h2, h3, p {
    margin: 0;
    padding: 0;
}

/* Rounded input fields */
.stSlider>div>div>div>div>input, .stNumberInput>div>input, .stSelectbox>div>div>div>select {
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------- Header -------------------------
st.markdown("""
<div class="header">
    <h1>🎓 Student Placement Prediction</h1>
    <p>Enter your academic and personal details below to predict placement status</p>
</div>
""", unsafe_allow_html=True)

# ------------------------- Main Container -------------------------
st.markdown('<div style="max-width:1200px; margin:auto; padding-top:20px;">', unsafe_allow_html=True)

# ------------------------- Input Section -------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="small")

with col1:
    st.markdown("<h3 style='color:#4B8BBE;'>📊 Academic Details</h3>", unsafe_allow_html=True)
    ssc_p = st.slider("SSC Percentage", 0.0, 100.0, 70.0, step=0.1)
    hsc_p = st.slider("HSC Percentage", 0.0, 100.0, 70.0, step=0.1)
    degree_p = st.slider("Degree Percentage", 0.0, 100.0, 70.0, step=0.1)
    etest_p = st.slider("E-test Percentage", 0.0, 100.0, 70.0, step=0.1)
    salary = st.number_input("Salary (if applicable, else 0)", min_value=0.0, value=0.0)

with col2:
    st.markdown("<h3 style='color:#4B8BBE;'>🧑 Personal Details</h3>", unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])
    workex = st.selectbox("Work Experience", ["Yes", "No"])
    degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])

st.markdown('</div>', unsafe_allow_html=True)  # close input card

# ------------------------- Load Model & Artifacts -------------------------
try:
    model = joblib.load("models/placement_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
except:
    st.error("❌ Model files not found. Please train the model first.")
    st.stop()

# ------------------------- Prepare Input DataFrame -------------------------
df_input = pd.DataFrame({
    'ssc_p': [ssc_p],
    'hsc_p': [hsc_p],
    'degree_p': [degree_p],
    'etest_p': [etest_p],
    'salary': [salary],
    'gender': [gender],
    'workex': [workex],
    'degree_t': [degree_t]
})

df_input['gender'] = df_input['gender'].map({'Male': 1, 'Female': 0})
df_input['workex_binary'] = df_input['workex'].map({'Yes': 1, 'No': 0})
df_input.drop('workex', axis=1, inplace=True)
df_input = pd.get_dummies(df_input, columns=['degree_t'], drop_first=True)

for col in feature_cols:
    if col not in df_input.columns:
        df_input[col] = 0
df_input = df_input[feature_cols]

num_cols_in_scaler = [col for col in scaler.feature_names_in_ if col in df_input.columns]
df_input[num_cols_in_scaler] = scaler.transform(df_input[num_cols_in_scaler])

# ------------------------- Prediction Section -------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)

pred_col1, pred_col2 = st.columns([3, 1], gap="small")
with pred_col1:
    st.markdown("<h3 style='color:#4B8BBE;'>📈 Your Placement Prediction</h3>", unsafe_allow_html=True)
with pred_col2:
    predict_btn = st.button("Predict Placement", type="primary")

if predict_btn:
    prediction = model.predict(df_input)[0]
    confidence = np.max(model.predict_proba(df_input)) * 100

    if prediction == 1:
        st.markdown(f"""
        <div style="background-color:#d4edda; padding:20px; border-radius:12px; text-align:center; box-shadow:0 4px 20px rgba(0,0,0,0.08);">
            <h2 style="color:#155724;">✅ PLACED!</h2>
            <p style="color:#155724; font-size:18px;">Confidence: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
        <div style="background-color:#fff3cd; padding:20px; border-radius:12px; text-align:center; box-shadow:0 4px 20px rgba(0,0,0,0.08);">
            <h2 style="color:#856404;">❌ NOT PLACED</h2>
            <p style="color:#856404; font-size:18px;">Confidence: {confidence:.2f}%</p>
            <p style="color:#856404;">Keep improving your skills and try again!</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close prediction card
st.markdown('</div>', unsafe_allow_html=True)  # close main container

# ------------------------- Footer -------------------------
st.markdown("""
<div class="footer">
    Developed with ❤️ by Harini Sri for student placement prediction
</div>
""", unsafe_allow_html=True)

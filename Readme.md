Student Placement Prediction

A Python-based project that predicts student placement outcomes using machine learning, and provides insightful visualizations and exploratory data analysis (EDA) for educational data.

📝 Project Overview

Many students and educational institutions struggle to identify the key factors influencing placement outcomes. This project uses historical student data to:

Predict whether a student will get placed

Visualize trends in placement data

Highlight key attributes affecting placement success

The project demonstrates end-to-end data science workflow: EDA → Feature Engineering → ML Modeling → Deployment (Streamlit).

💡 Problem Statement

Predict the placement status of students (Placed / Not Placed) based on academic and demographic features.

Provide insights into how factors like degree, specialization, work experience, and academic performance affect placements.

Help students and institutions identify key predictors of employability.

📂 Project Structure
Student_Placement_Prediction/
│
├── data/                # Input datasets (CSV files)
│   ├── Placement_Data_Full_Class.csv
│   └── 03_EDA_Cleaned.csv
│
├── scripts/             # Python scripts for processing and modeling
│   ├── 01_load_data.py
│   ├── 02_data_cleaning.py
│   ├── 03_EDA.py
│   ├── 04_Categorical_Visuals.py
│   ├── 05_Data_Preprocessing.py
│   └── 06_Modeling.py
│
├── models/              # Trained ML models
│   ├── placement_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── outputs/             # Generated visualizations
│   ├── placement_by_gender.png
│   ├── salary_distribution.png
│   └── workex_vs_placement.png
│
├── app.py               # Streamlit interactive prediction app
├── README.md            # Project documentation
└── EDA_INSIGHTS.md      # Detailed EDA report

🛠 Tech Stack

Python – Data processing, analysis, and ML modeling

Pandas & NumPy – Data manipulation

Matplotlib & Seaborn – Visualization

Scikit-learn – ML models (Logistic Regression, Random Forest)

Streamlit – Interactive web app

📊 Key Features

Cleaned and preprocessed student placement dataset

Exploratory Data Analysis (EDA) with graphs and correlation heatmaps

Predictive ML models for placement outcome

Interactive Streamlit app for real-time predictions

🚀 How to Run Locally

Clone the repository:

git clone https://github.com/<your-username>/Student_Placement_Prediction.git
cd Student_Placement_Prediction


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py


Open the browser link provided by Streamlit to interact with the app.

📈 Results & Insights

Key factors affecting placement identified: gender, specialization, work experience, degree, and academic scores

Machine learning models predict placement accurately

Visualizations provide actionable insights for students and institutions

🎯 Impact

Helps students understand factors affecting employability

Enables institutions to improve placement outcomes

Demonstrates full end-to-end data science workflow

# AI-powered diabetes prediction system using Python, Flask and React.js - Kiran Veerabatheni 
#Diabetes prediction using Random Forest model on dataset from National Institute of Diabetes and Digestive and Kidney Diseases (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data)
#* @AppliedAIwithKiran - kiran.veerabatheni@hotmail.com
# 
📌 Overview
This project demonstrates an end-to-end AI-powered diabetes prediction system using the Pima Indians Diabetes Dataset from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK). The solution includes:

Machine Learning Model Development (Python, scikit-learn)

REST API Deployment (Flask, Waitress)

Interactive Web Dashboard (React, Axios)

Early diabetes detection can significantly improve patient outcomes, and this implementation showcases how AI can be integrated into real-world healthcare applications.

🔧 Project Structure
The repository is organized into three main components:

/model – Contains the Python script for data preprocessing, model training (Random Forest), evaluation, and serialization (Pickle).

/api – Flask-based REST API for model inference, deployed using Waitress for production.

/react-app – Frontend React application that consumes the API and displays predictions in a dashboard.

🚀 Key Features
1. Model Development (k-diabetes-prediction-main.py)
Data Cleaning & EDA: Handles missing values, visualizes feature distributions, and generates correlation matrices.

Random Forest Classifier: Trained with hyperparameter tuning (GridSearchCV) for optimal performance.

Model Evaluation: Metrics include accuracy, precision, recall, F1-score, and ROC-AUC.

Pickle Serialization: Saves the trained model, scaler, and training data for deployment.

2. API Deployment (k-diabetes-prediction-api.py)
Flask REST API: Exposes a /predict endpoint for real-time inferencing.

Scaler Integration: Ensures input data is normalized before prediction.

Waitress Production Server: Enables high-performance deployment.

Optional Windows Service (NSSM): For 24/7 reliability.

Swagger Documentation: Auto-generated API docs for easy testing.

3. Frontend Integration (React + Axios)
User-Friendly Dashboard: Collects patient biomarkers via form inputs.

Real-Time Predictions: Displays diabetes risk with confidence scores.

Visualizations: Comparative charts against training data.

⚙️ Technology Stack
Backend: Python, Flask, scikit-learn, Waitress

Frontend: React, TypeScript, Axios

Deployment: Waitress, NSSM (Windows Service)

📂 How to Run
1. Model Training
bash
Copy
cd model
python k-diabetes-prediction-main.py
(Requires: pandas, scikit-learn, matplotlib, seaborn)

2. API Deployment
bash
Copy
cd api
python k-diabetes-prediction-api.py
(API runs on http://0.0.0.0:8800 by default)

3. Frontend (React App)
bash
Copy
cd react-app
npm install
npm start
(Configure apiUrl in App.tsx to match your Flask server)

📊 Sample Outputs
Model Performance Metrics (Accuracy, F1-score, etc.)

Feature Importance & EDA Visualizations

API Response Example:

json
Copy
{
  "prediction": 1,
  "probability": 0.87,
  "diabetes_status": "Positive",
  "visualizations": { ... }
}
🌐 Use Case in Healthcare
This system assists doctors in early diabetes detection by analyzing key biomarkers (glucose, BMI, insulin levels, etc.). It does not replace medical expertise but enhances decision-making with AI-driven insights.

📜 License
MIT

🔗 GitHub Repo: https://github.com/kiranvsamuel/k-ml-diabetes-detection

#--------------------------------------------------------------------------

kiran.veerabatheni@hotmail.com

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

print("🔄 STAGE 1: Initializing Flask Application context...")
app = Flask(__name__)

# Check if model files are physically visible to this file
print(f"📁 STAGE 2: Scanning working directory... Current path: {os.getcwd()}")
print(f"   └── Files found: {os.listdir('.')}")

diagnostic_model = None
medical_scaler = None

# Wrap loading in a verbose try-except block to trap silent crashes
try:
    print("🧠 STAGE 3: Loading Scikit-Learn/XGBoost Ensemble weights...")
    diagnostic_model = joblib.load('disease_ensemble_model.pkl')
    print("✅ Ensemble weights loaded perfectly!")
    
    print("📏 STAGE 4: Loading Standard Scaler transformation keys...")
    medical_scaler = joblib.load('disease_scaler.pkl')
    print("✅ Scaler metrics loaded perfectly!")
except Exception as error_msg:
    print(f"❌ CRITICAL LOAD FAILURE: The pipeline crashed while reading the .pkl files.")
    print(f"👉 ERROR DETAIL: {str(error_msg)}")
    print("💡 Fix: If this failed, your Colab python version doesn't match your local machine version.")

@app.route('/')
def home():
    print("🌐 Browser requested the home path '/' - Rendering templates/index.html")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = float(request.form['trestbps'])
            chol = float(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = float(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            patient_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=patient_features)

            num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            input_df[num_cols] = medical_scaler.transform(input_df[num_cols])

            prediction = diagnostic_model.predict(input_df)[0]
            risk_probability = diagnostic_model.predict_proba(input_df)[0][1] * 100

            if prediction == 0:
                result_text = "APPROVED: LOW RISK CARDIO PROFILE DETECTED"
                result_class = "success"
                details = f"The patient vitals fall safely within traditional health boundaries. Ensemble Confidence: {100 - risk_probability:.2f}% Safe."
            else:
                result_text = "ALERT: CARDIOVASCULAR DISEASE INDICATORS FOUND"
                result_class = "danger"
                details = f"Critical Threshold Crossed. The patient patterns align strongly with historical disease cohorts. Ensemble Risk Factor: {risk_probability:.2f}% Volatile."

            return render_template('index.html', prediction_text=result_text, result_class=result_class, details_text=details)

        except Exception as e:
            return render_template('index.html', prediction_text=f"Pipeline Processing Error: {str(e)}", result_class="danger")

# REMOVED THE IF-NAME GUARD TO FORCE FLASK RUN IRRESPECTIVE OF CONTEXT
print("🚀 STAGE 5: Bypassing structural guards. Launching Local Server...")
app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
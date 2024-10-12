from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load models and scaler
scaler = joblib.load('Models/scaler.pkl')
adaboost_svm = joblib.load('Models/adaboost_svm.pkl')
adaboost_dt = joblib.load('Models/adaboost_dt.pkl')
adaboost_lr = joblib.load('Models/adaboost_lr.pkl')
adaboost_rf = joblib.load('Models/adaboost_rf.pkl')

# Prediction function
def predict_breast_cancer(features):
    # Convert the input features to a DataFrame
    input_data = pd.DataFrame(features)
    # Scale the input features
    input_scaled = scaler.transform(input_data)

    # Make predictions with each trained AdaBoost model
    pred_svm = adaboost_svm.predict(input_scaled)
    pred_dt = adaboost_dt.predict(input_scaled)
    pred_lr = adaboost_lr.predict(input_scaled)
    pred_rf = adaboost_rf.predict(input_scaled)

    # Combine predictions using majority voting
    ensemble_prediction = [max(set(pred), key=pred.count) for pred in zip(pred_svm, pred_dt, pred_lr, pred_rf)]

    # Decode the numeric predictions back to 'B' (benign) and 'M' (malignant)
    prediction_result = ['B' if pred == 0 else 'M' for pred in ensemble_prediction]

    return prediction_result[0]  # Return the first prediction

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction input page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get features from form
        features = {
            'radius_mean': [float(request.form['radius_mean'])],
            'texture_mean': [float(request.form['texture_mean'])],
            'perimeter_mean': [float(request.form['perimeter_mean'])],
            'area_mean': [float(request.form['area_mean'])],
            'smoothness_mean': [float(request.form['smoothness_mean'])],
            'compactness_mean': [float(request.form['compactness_mean'])],
            'concavity_mean': [float(request.form['concavity_mean'])],
            'concave points_mean': [float(request.form['concave_points_mean'])],
            'symmetry_mean': [float(request.form['symmetry_mean'])],
            'fractal_dimension_mean': [float(request.form['fractal_dimension_mean'])],
            'radius_se': [float(request.form['radius_se'])],
            'texture_se': [float(request.form['texture_se'])],
            'perimeter_se': [float(request.form['perimeter_se'])],
            'area_se': [float(request.form['area_se'])],
            'smoothness_se': [float(request.form['smoothness_se'])],
            'compactness_se': [float(request.form['compactness_se'])],
            'concavity_se': [float(request.form['concavity_se'])],
            'concave points_se': [float(request.form['concave_points_se'])],
            'symmetry_se': [float(request.form['symmetry_se'])],
            'fractal_dimension_se': [float(request.form['fractal_dimension_se'])],
            'radius_worst': [float(request.form['radius_worst'])],
            'texture_worst': [float(request.form['texture_worst'])],
            'perimeter_worst': [float(request.form['perimeter_worst'])],
            'area_worst': [float(request.form['area_worst'])],
            'smoothness_worst': [float(request.form['smoothness_worst'])],
            'compactness_worst': [float(request.form['compactness_worst'])],
            'concavity_worst': [float(request.form['concavity_worst'])],
            'concave points_worst': [float(request.form['concave_points_worst'])],
            'symmetry_worst': [float(request.form['symmetry_worst'])],
            'fractal_dimension_worst': [float(request.form['fractal_dimension_worst'])],
        }
        # Get prediction
        prediction = predict_breast_cancer(features)
        return render_template('result.html', prediction=prediction)
    return render_template('predict.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
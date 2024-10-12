import pandas as pd
import joblib

# Load the saved scaler and models
scaler = joblib.load('Models/scaler.pkl')
adaboost_svm = joblib.load('Models/adaboost_svm.pkl')
adaboost_dt = joblib.load('Models/adaboost_dt.pkl')
adaboost_lr = joblib.load('Models/adaboost_lr.pkl')
adaboost_rf = joblib.load('Models/adaboost_rf.pkl')

# Function to make predictions
def predict_breast_cancer(features):
    # Convert the input features to a DataFrame
    input_data = pd.DataFrame(features)

    # Scale the input features using the previously fitted scaler
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

    return prediction_result

# Example usage
if __name__ == "__main__":
    # Input features for prediction
    features = {
        'radius_mean': [float(input("Enter radius_mean: "))],
        'texture_mean': [float(input("Enter texture_mean: "))],
        'perimeter_mean': [float(input("Enter perimeter_mean: "))],
        'area_mean': [float(input("Enter area_mean: "))],
        'smoothness_mean': [float(input("Enter smoothness_mean: "))],
        'compactness_mean': [float(input("Enter compactness_mean: "))],
        'concavity_mean': [float(input("Enter concavity_mean: "))],
        'concave points_mean': [float(input("Enter concave points_mean: "))],
        'symmetry_mean': [float(input("Enter symmetry_mean: "))],
        'fractal_dimension_mean': [float(input("Enter fractal_dimension_mean: "))],
        'radius_se': [float(input("Enter radius_se: "))],
        'texture_se': [float(input("Enter texture_se: "))],
        'perimeter_se': [float(input("Enter perimeter_se: "))],
        'area_se': [float(input("Enter area_se: "))],
        'smoothness_se': [float(input("Enter smoothness_se: "))],
        'compactness_se': [float(input("Enter compactness_se: "))],
        'concavity_se': [float(input("Enter concavity_se: "))],
        'concave points_se': [float(input("Enter concave points_se: "))],
        'symmetry_se': [float(input("Enter symmetry_se: "))],
        'fractal_dimension_se': [float(input("Enter fractal_dimension_se: "))],
        'radius_worst': [float(input("Enter radius_worst: "))],
        'texture_worst': [float(input("Enter texture_worst: "))],
        'perimeter_worst': [float(input("Enter perimeter_worst: "))],
        'area_worst': [float(input("Enter area_worst: "))],
        'smoothness_worst': [float(input("Enter smoothness_worst: "))],
        'compactness_worst': [float(input("Enter compactness_worst: "))],
        'concavity_worst': [float(input("Enter concavity_worst: "))],
        'concave points_worst': [float(input("Enter concave points_worst: "))],
        'symmetry_worst': [float(input("Enter symmetry_worst: "))],
        'fractal_dimension_worst': [float(input("Enter fractal_dimension_worst: "))],
    }

    # Get the prediction
    result = predict_breast_cancer(features)
    print(f'Predicted diagnosis: {result[0]}')  # Output the prediction

'''sample input:
88995002	M	20.73	31.12	135.7	1419	0.09469	0.1143	0.1367	0.08646	0.1769	0.05674	1.172	1.617	7.749	199.7	0.004551	0.01478	0.02143	0.00928	0.01367	0.002299	32.49	47.16	214	3432	0.1401	0.2644	0.3442	0.1659	0.2868	0.08218	
8910251	B	10.6	18.95	69.28	346.4	0.09688	0.1147	0.06387	0.02642	0.1922	0.06491	0.4505	1.197	3.43	27.1	0.00747	0.03581	0.03354	0.01365	0.03504	0.003318	11.88	22.94	78.28	424.8	0.1213	0.2515	0.1916	0.07926	0.294	0.07587	
''' 
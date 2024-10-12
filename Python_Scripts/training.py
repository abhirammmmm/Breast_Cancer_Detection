import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib  # For saving models
import warnings  # Import warnings module

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv('Data/data.csv')
print(data.head())  # Display the first few rows of the dataset

# Check the data types of each column
print(data.dtypes)

# Remove unnecessary columns
data.drop(columns=['Unnamed: 32'], inplace=True)  # Drop the column if it contains only NaNs or is not needed

# Handle non-numeric columns
# Convert 'diagnosis' to numeric using Label Encoding
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Check for NaN values and remove any rows or columns with NaN values
print("Missing values before handling:", data.isnull().sum())
data.fillna(data.mean(numeric_only=True), inplace=True)  # Fill NaNs with the mean for numeric columns
print("Missing values after handling:", data.isnull().sum())

# Define features and target
X = data.drop(columns=['id', 'diagnosis'])  # Features: all columns except 'id' and 'diagnosis'
y = data['diagnosis']  # Target variable: 'diagnosis'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit the scaler on the training data and transform it
X_test_scaled = scaler.transform(X_test)  # Transform the test data using the same scaler

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')

# Create base classifiers
svm_classifier = SVC(probability=True, kernel='linear', random_state=42)
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Hyperparameter tuning for SVM
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
}
svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=5)
svm_grid.fit(X_train_scaled, y_train)
best_svm_classifier = svm_grid.best_estimator_

# Create AdaBoost classifiers for each base estimator
adaboost_svm = AdaBoostClassifier(estimator=best_svm_classifier, n_estimators=50, random_state=42)
adaboost_dt = AdaBoostClassifier(estimator=dt_classifier, n_estimators=50, random_state=42)
adaboost_lr = AdaBoostClassifier(estimator=lr_classifier, n_estimators=50, random_state=42)
adaboost_rf = AdaBoostClassifier(estimator=rf_classifier, n_estimators=50, random_state=42)

# Train each AdaBoost classifier
adaboost_svm.fit(X_train_scaled, y_train)
adaboost_dt.fit(X_train_scaled, y_train)
adaboost_lr.fit(X_train_scaled, y_train)
adaboost_rf.fit(X_train_scaled, y_train)

# Save each trained model to .pkl files
joblib.dump(adaboost_svm, 'Models/adaboost_svm.pkl')
joblib.dump(adaboost_dt, 'Models/adaboost_dt.pkl')
joblib.dump(adaboost_lr, 'Models/adaboost_lr.pkl')
joblib.dump(adaboost_rf, 'Models/adaboost_rf.pkl')

# Cross-validation for performance evaluation
svm_scores = cross_val_score(adaboost_svm, X_train_scaled, y_train, cv=5)
dt_scores = cross_val_score(adaboost_dt, X_train_scaled, y_train, cv=5)
lr_scores = cross_val_score(adaboost_lr, X_train_scaled, y_train, cv=5)
rf_scores = cross_val_score(adaboost_rf, X_train_scaled, y_train, cv=5)

print("Cross-Validation Accuracy for SVM:", svm_scores.mean())
print("Cross-Validation Accuracy for Decision Tree:", dt_scores.mean())
print("Cross-Validation Accuracy for Logistic Regression:", lr_scores.mean())
print("Cross-Validation Accuracy for Random Forest:", rf_scores.mean())

# Make predictions with each classifier
y_pred_svm = adaboost_svm.predict(X_test_scaled)
y_pred_dt = adaboost_dt.predict(X_test_scaled)
y_pred_lr = adaboost_lr.predict(X_test_scaled)
y_pred_rf = adaboost_rf.predict(X_test_scaled)

# Combine predictions (e.g., majority vote)
ensemble_predictions = [max(set(pred), key=pred.count) for pred in zip(y_pred_svm, y_pred_dt, y_pred_lr, y_pred_rf)]

# Calculate accuracy
accuracy = accuracy_score(y_test, ensemble_predictions)
print(f'Accuracy on Test Set: {accuracy:.2f}')

# Calculate precision, recall, and F1-score for each classifier
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Calculate precision, recall, and F1-score for AdaBoost classifier
precision_adaboost = precision_score(y_test, ensemble_predictions)
recall_adaboost = recall_score(y_test, ensemble_predictions)
f1_adaboost = f1_score(y_test, ensemble_predictions)

# Print precision, recall, and F1-score for each classifier
print("SVM Precision:", precision_svm)
print("SVM Recall:", recall_svm)
print("SVM F1-score:", f1_svm)

print("Decision Tree Precision:", precision_dt)
print("Decision Tree Recall:", recall_dt)
print("Decision Tree F1-score:", f1_dt)

print("Logistic Regression Precision:", precision_lr)
print("Logistic Regression Recall:", recall_lr)
print("Logistic Regression F1-score:", f1_lr)

print("Random Forest Precision:", precision_rf)
print("Random Forest Recall:", recall_rf)
print("Random Forest F1-score:", f1_rf)

# Print precision, recall, and F1-score for AdaBoost classifier
print("AdaBoost Precision:", precision_adaboost)
print("AdaBoost Recall:", recall_adaboost)
print("AdaBoost F1-score:", f1_adaboost)
# ===============================
# SMART AQI SYSTEM - MODEL TRAINING
# ===============================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("data/air_quality_data.csv")

# Convert Date
df['Date'] = pd.to_datetime(df['Date'])

# Drop rows where AQI is missing
df = df.dropna(subset=['AQI'])

# Fill remaining missing values using median
df = df.fillna(df.median(numeric_only=True))

# Encode City
le = LabelEncoder()
df['City'] = le.fit_transform(df['City'])

# Drop AQI_Bucket for regression
df = df.drop(columns=['AQI_Bucket'], errors='ignore')

# Define features and target
X = df.drop(columns=['AQI', 'Date'])
y = df['AQI']

print("Final Dataset Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("AQI Min:", df['AQI'].min())
print("AQI Max:", df['AQI'].max())

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestRegressor(random_state=42)

# Hyperparameter Tuning
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Stage-2 Tuned Model RMSE:", rmse)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Save model
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
print("Model and feature columns saved successfully!")

# Save label encoder
joblib.dump(le, "models/city_encoder.pkl")
# Save unique city names
joblib.dump(df['City'].unique().tolist(), "models/city_encoded_values.pkl")
print("Model, features, and encoder saved successfully!")

metrics = {
    "RMSE": float(rmse),
    "R2": float(r2)
}

joblib.dump(metrics, "models/model_metrics.pkl")



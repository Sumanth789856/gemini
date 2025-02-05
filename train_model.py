import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("crop_data.csv")

# Split features and labels
X = data[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = data["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model to a file
with open("models/crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")

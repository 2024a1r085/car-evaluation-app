import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

df = pd.read_csv("car.data", names=columns)

# Encode data
encoders = {}

for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and target
X = df.drop("class", axis=1)
y = df["class"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model + encoders
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Model saved!")

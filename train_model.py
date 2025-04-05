
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("your_dataset.csv")  # Replace with your actual file name
X = data.drop("Target", axis=1)
y = data["Target"]

# One-hot encode the categorical input features
X_encoded = pd.get_dummies(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and column names
joblib.dump(model, "random_forest_model.pkl")
joblib.dump(X_train.columns.tolist(), "model_columns.pkl")
print("✅ Model and columns saved successfully!")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# --- CONFIGURATION ---
root = os.getcwd()
DATASET_PATH = os.path.join(root,'keypoints.csv')
MODEL_PATH = 'model.pkl'

# 1. Load Data
# We manually define names because our CSV doesn't have a header row
# 21 points * 2 (x,y) = 42 columns + 1 label column = 43 columns
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH, header=None)

# Separate features (X) and labels (y)
X = df.iloc[:, 1:].values  # All columns except the first one (coordinates)
y = df.iloc[:, 0].values   # The first column (label)

# 2. Split into Training and Testing sets
# 80% for training, 20% for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define the Neural Network Classifier
# Hidden_layer_sizes=(128, 64) means:
#   - Input Layer: 42 neurons (implicit)
#   - Hidden Layer 1: 128 neurons
#   - Hidden Layer 2: 64 neurons
#   - Output Layer: Number of classes (implicit)
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    verbose=True # Prints progress
)

# 4. Train the Model
print("Training model...")
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save the Model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved successfully to {MODEL_PATH}")
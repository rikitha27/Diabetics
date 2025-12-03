# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# 1. Load data (change path if needed)
df = pd.read_csv("diabetes.csv")  # keep dataset in the same folder

# 2. Preprocessing: replace 0 with NaN for some columns and fill with mean
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

# 3. Split features & target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Save model + column order
artifact = {
    "model": model,
    "columns": X.columns.tolist()
}

with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("Model saved to diabetes_model.pkl")

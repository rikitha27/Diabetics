from flask import Flask, render_template, request, flash
import numpy as np
import pickle
import os

app = Flask(__name__)
app.secret_key = "super_secret_key_change_me"  # needed for flash messages

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")

# Load model + column order
with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
feature_columns = artifact["columns"]  # list of column names in correct order

# Pretty labels for UI
LABELS = {
    "Pregnancies": "Number of Pregnancies",
    "Glucose": "Glucose (mg/dL)",
    "BloodPressure": "Blood Pressure (mm Hg)",
    "SkinThickness": "Skin Thickness (mm)",
    "Insulin": "Insulin (mu U/ml)",
    "BMI": "Body Mass Index (BMI)",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age (years)"
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    input_values = {}

    if request.method == "POST":
        try:
            values = []
            for col in feature_columns:
                raw_val = request.form.get(col, "").strip()
                input_values[col] = raw_val  # keep for redisplay
                if raw_val == "":
                    raise ValueError("Empty value")
                values.append(float(raw_val))

            input_array = np.array([values])

            # Predict class
            pred_class = model.predict(input_array)[0]
            prediction = "Diabetic" if pred_class == 1 else "Not Diabetic"

            # Predict probability (if available)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_array)[0][1]
                probability = round(proba * 100, 2)

        except ValueError:
            flash("Please enter valid numeric values for all fields.")

    return render_template(
        "index.html",
        feature_columns=feature_columns,
        labels=LABELS,
        prediction=prediction,
        probability=probability,
        input_values=input_values
    )


# Optional: simple JSON API endpoint
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    try:
        values = [float(data[col]) for col in feature_columns]
    except Exception:
        return {"error": "Invalid or missing values"}, 400

    input_array = np.array([values])
    pred_class = model.predict(input_array)[0]
    prediction = "Diabetic" if pred_class == 1 else "Not Diabetic"

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_array)[0][1]

    return {
        "prediction": prediction,
        "class": int(pred_class),
        "probability": proba
    }


if __name__ == "__main__":
    app.run(debug=True)

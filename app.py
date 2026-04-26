"""
Talk to Me — Telecom Churn Prediction API
Flask backend that serves the dashboard and prediction endpoints.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore")

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # 5 MB upload limit

# ── Load model artefacts ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model            = joblib.load(os.path.join(MODELS_DIR, "tuned_churn_model.pkl"))
scaler           = joblib.load(os.path.join(MODELS_DIR, "churn_scaler.pkl"))
selected_features = joblib.load(os.path.join(MODELS_DIR, "selected_features.pkl"))

with open(os.path.join(MODELS_DIR, "model_config.json")) as f:
    config = json.load(f)

THRESHOLD     = config.get("optimal_threshold", 0.40)
MODEL_NAME    = config.get("model_name", "XGBoost")
VAL_AUC       = config.get("val_auc", 0.0)

print(f"✅ Model loaded: {MODEL_NAME}  |  threshold={THRESHOLD}  |  AUC={VAL_AUC}")
print(f"   Features used: {selected_features}")

# ── Feature engineering helpers ───────────────────────────────────────────────
# Maps the exact keys sent from the form to the encoded feature names
# This must mirror the encoding done during training exactly.

BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

def encode_row(raw: dict) -> dict:
    """
    Convert a raw form/JSON dict → encoded feature dict
    matching the training pipeline.
    """
    r = {}

    # Binary fields
    for col in ["gender", "Partner", "Dependents", "PhoneService",
                "PaperlessBilling", "MultipleLines", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies"]:
        val = str(raw.get(col, "No"))
        # Normalise "No internet/phone service" → 0
        r[col] = 1 if val == "Yes" else 0

    # Numeric fields
    r["SeniorCitizen"]  = int(raw.get("SeniorCitizen", 0))
    r["tenure"]         = float(raw.get("tenure", 0))
    r["MonthlyCharges"] = float(raw.get("MonthlyCharges", 0))
    r["TotalCharges"]   = float(raw.get("TotalCharges", 0))

    # InternetService one-hot (drop_first → DSL is baseline)
    internet = str(raw.get("InternetService", "DSL"))
    r["InternetService_Fiber optic"] = 1 if internet == "Fiber optic" else 0
    r["InternetService_No"]          = 1 if internet == "No" else 0

    # Contract one-hot (drop_first → Month-to-month is baseline)
    contract = str(raw.get("Contract", "Month-to-month"))
    r["Contract_One year"] = 1 if contract == "One year" else 0
    r["Contract_Two year"] = 1 if contract == "Two year" else 0

    # PaymentMethod one-hot (drop_first → Bank transfer is baseline)
    payment = str(raw.get("PaymentMethod", "Bank transfer (automatic)"))
    r["PaymentMethod_Credit card (automatic)"] = 1 if payment == "Credit card (automatic)" else 0
    r["PaymentMethod_Electronic check"]        = 1 if payment == "Electronic check" else 0
    r["PaymentMethod_Mailed check"]            = 1 if payment == "Mailed check" else 0

    return r


def predict_from_df(df_encoded: pd.DataFrame):
    """Scale and predict a DataFrame that already has encoded columns."""
    # Keep only features the model was trained on, in correct order
    X = df_encoded[selected_features].values
    X_scaled = scaler.transform(X)
    probas    = model.predict_proba(X_scaled)[:, 1]
    preds     = (probas >= THRESHOLD).astype(int)
    return probas, preds


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           model_name=MODEL_NAME,
                           val_auc=VAL_AUC,
                           threshold=THRESHOLD)


@app.route("/predict", methods=["POST"])
def predict_single():
    """Predict churn for a single customer from JSON body."""
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data received"}), 400

    try:
        encoded = encode_row(data)
        df      = pd.DataFrame([encoded])
        probas, preds = predict_from_df(df)

        probability = float(probas[0])
        prediction  = int(preds[0])
        risk_level  = (
            "High Risk"   if probability >= 0.65 else
            "Medium Risk" if probability >= 0.40 else
            "Low Risk"
        )

        return jsonify({
            "prediction":  prediction,
            "probability": round(probability * 100, 1),
            "risk_level":  risk_level,
            "will_churn":  bool(prediction == 1),
            "threshold":   THRESHOLD
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """Predict churn for a CSV file upload."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    try:
        df_raw = pd.read_csv(file)

        # ── Replicate training pre-processing ────────────────────────────
        # 1. Drop customerID if present (save it for display)
        customer_ids = df_raw.get("customerID", pd.Series(range(len(df_raw)), name="customerID"))
        df = df_raw.drop(columns=["customerID"], errors="ignore").copy()

        # 2. TotalCharges → float
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

        # 3. Drop Churn column if present (it's the target, not an input)
        df = df.drop(columns=["Churn"], errors="ignore")

        # 4. Encode each row
        encoded_rows = [encode_row(row.to_dict()) for _, row in df.iterrows()]
        df_encoded   = pd.DataFrame(encoded_rows)

        # 5. Predict
        probas, preds = predict_from_df(df_encoded)

        results = []
        for i, (prob, pred) in enumerate(zip(probas, preds)):
            risk = ("High Risk"   if prob >= 0.65 else
                    "Medium Risk" if prob >= 0.40 else
                    "Low Risk")
            results.append({
                "customer_id":  str(customer_ids.iloc[i]) if i < len(customer_ids) else str(i + 1),
                "probability":  round(float(prob) * 100, 1),
                "prediction":   int(pred),
                "will_churn":   bool(pred == 1),
                "risk_level":   risk
            })

        # Summary stats
        total        = len(results)
        churners     = sum(r["will_churn"] for r in results)
        avg_prob     = round(float(probas.mean()) * 100, 1)
        high_risk    = sum(1 for r in results if r["risk_level"] == "High Risk")
        medium_risk  = sum(1 for r in results if r["risk_level"] == "Medium Risk")
        low_risk     = sum(1 for r in results if r["risk_level"] == "Low Risk")

        return jsonify({
            "results":     results,
            "summary": {
                "total":       total,
                "churners":    churners,
                "retained":    total - churners,
                "churn_rate":  round(churners / total * 100, 1),
                "avg_prob":    avg_prob,
                "high_risk":   high_risk,
                "medium_risk": medium_risk,
                "low_risk":    low_risk
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "auc": VAL_AUC})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

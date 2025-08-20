from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("loanstatus_model.h5")

# Load saved scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get form data in the same order as training
            married = 1 if request.form.get("married") == "Yes" else 0
            dependents = int(request.form.get("dependents", 0))
            education = 1 if request.form.get("education") == "Graduate" else 0
            coapplicantincome = float(request.form.get("coapplicantincome", 0))
            loanamount = float(request.form.get("loanamount", 0))
            loan_amount_term = float(request.form.get("loanamountterm", 0))
            credit_history = float(request.form.get("credit_history", 0))
            property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}.get(request.form.get("property_area"), 0)

            # Arrange in the same order as training
            features = np.array([[married, dependents, education,
                                   coapplicantincome, loanamount,
                                   loan_amount_term, credit_history,
                                   property_area]])

            # Apply scaling
            features_scaled = scaler.transform(features)

            # Predict
            pred = model.predict(features_scaled)
            prediction = "Approved" if pred[0][0] > 0.5 else "Rejected"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

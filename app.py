import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flask import Flask, render_template, request
import pandas as pd

from mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__)


@app.route("/", methods=["GET"])
def homePage():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # ---------------------------------
        # Get complete submitted form
        # ---------------------------------

        form_data = request.form.to_dict()

        print("\n================================")
        print("FORM DATA RECEIVED")
        print("================================")
        print(form_data)
        print("================================\n")

        # ---------------------------------
        # Required fields
        # ---------------------------------

        required_fields = [
            "age",
            "income",
            "loan_amount",
            "credit_score",
            "months_employed",
            "num_credit_lines",
            "interest_rate",
            "loan_term",
            "dti_ratio",
            "education",
            "employment_type",
            "marital_status",
            "has_mortgage",
            "has_dependents",
            "loan_purpose",
            "has_cosigner"
        ]

        # Check missing fields safely
        missing_fields = [
            field
            for field in required_fields
            if not form_data.get(field)
        ]

        if missing_fields:
            print("MISSING FIELDS:", missing_fields)

            return (
                "Missing fields: " + ", ".join(missing_fields),
                400
            )

        # ---------------------------------
        # Numerical values
        # ---------------------------------

        age = int(form_data["age"])
        income = int(form_data["income"])
        loan_amount = int(form_data["loan_amount"])
        credit_score = int(form_data["credit_score"])
        months_employed = int(form_data["months_employed"])
        num_credit_lines = int(form_data["num_credit_lines"])

        interest_rate = float(form_data["interest_rate"])
        loan_term = int(form_data["loan_term"])
        dti_ratio = float(form_data["dti_ratio"])

        # ---------------------------------
        # Categorical values
        # ---------------------------------

        education = form_data["education"]
        employment_type = form_data["employment_type"]
        marital_status = form_data["marital_status"]
        has_mortgage = form_data["has_mortgage"]
        has_dependents = form_data["has_dependents"]
        loan_purpose = form_data["loan_purpose"]
        has_cosigner = form_data["has_cosigner"]

        # ---------------------------------
        # Create input DataFrame
        # ---------------------------------

        data = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "MonthsEmployed": months_employed,
            "NumCreditLines": num_credit_lines,
            "InterestRate": interest_rate,
            "LoanTerm": loan_term,
            "DTIRatio": dti_ratio,
            "Education": education,
            "EmploymentType": employment_type,
            "MaritalStatus": marital_status,
            "HasMortgage": has_mortgage,
            "HasDependents": has_dependents,
            "LoanPurpose": loan_purpose,
            "HasCoSigner": has_cosigner
        }])

        print("\n================================")
        print("MODEL INPUT")
        print("================================")
        print(data)
        print("================================\n")

        # ---------------------------------
        # Prediction
        # ---------------------------------

        prediction_pipeline = PredictionPipeline()

        prediction = prediction_pipeline.predict(data)

        prediction_value = int(prediction[0])

        print("PREDICTION:", prediction_value)

        # ---------------------------------
        # Result
        # ---------------------------------

        if prediction_value == 0:
            result = "No Default Predicted"
            risk_level = "Low Default Risk"
        else:
            result = "Default Risk Detected"
            risk_level = "High Default Risk"

        return render_template(
            "results.html",
            prediction=prediction_value,
            result=result,
            risk_level=risk_level
        )

    except Exception as e:

        print("\n================================")
        print("PREDICTION ERROR")
        print("================================")
        print(repr(e))
        print("================================\n")

        return f"Prediction error: {e}", 500


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
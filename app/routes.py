# app/routes.py
from flask import Blueprint, render_template, request, jsonify
from . import utils

main = Blueprint("main", __name__, template_folder="templates")

@main.route("/", methods=["GET"])
def home():
    return render_template("index.html", title="FinSight")

@main.route("/loan-approval", methods=["GET", "POST"])
def loan_approval():
    result = None
    error = None
    if request.method == "POST":
        form_data = {
            "Gender": request.form.get("Gender"),
            "Married": request.form.get("Married"),
            "Dependents": request.form.get("Dependents"),
            "Education": request.form.get("Education"),
            "Self_Employed": request.form.get("Self_Employed"),
            "Credit_History": request.form.get("Credit_History"),
            "Property_Area": request.form.get("Property_Area"),
            "ApplicantIncome": request.form.get("ApplicantIncome"),
            "CoapplicantIncome": request.form.get("CoapplicantIncome"),
            "LoanAmount": request.form.get("LoanAmount"),
            "Loan_Amount_Term": request.form.get("Loan_Amount_Term")
        }
        result = utils.predict_approval(form_data)
        if not result.get("ok"):
            error = result.get("error")
    return render_template("loan_approval.html", title="Loan Approval", result=result, error=error)

@main.route("/loan-default", methods=["GET", "POST"])
def loan_default():
    result = None
    error = None
    if request.method == "POST":
        form_data = {
            "RevolvingUtilizationOfUnsecuredLines": request.form.get("RevolvingUtilizationOfUnsecuredLines"),
            "age": request.form.get("age"),
            "NumberOfTime30-59DaysPastDueNotWorse": request.form.get("NumberOfTime30-59DaysPastDueNotWorse"),
            "DebtRatio": request.form.get("DebtRatio"),
            "MonthlyIncome": request.form.get("MonthlyIncome"),
            "NumberOfOpenCreditLinesAndLoans": request.form.get("NumberOfOpenCreditLinesAndLoans"),
            "NumberOfTimes90DaysLate": request.form.get("NumberOfTimes90DaysLate"),
            "NumberRealEstateLoansOrLines": request.form.get("NumberRealEstateLoansOrLines"),
            "NumberOfTime60-89DaysPastDueNotWorse": request.form.get("NumberOfTime60-89DaysPastDueNotWorse"),
            "NumberOfDependents": request.form.get("NumberOfDependents"),
        }
        result = utils.predict_default(form_data, threshold=0.3)
        if not result.get("ok"):
            error = result.get("error")
    return render_template("loan_default.html", title="Loan Default", result=result, error=error)

@main.route("/debug-models", methods=["GET"])
def debug_models():
    """
    Debug endpoint: returns info about scaler.feature_names_in_ and model.feature_names.
    Use this if the app says there's a mismatch so we can inspect what the saved artifacts expect.
    """
    info = utils.get_model_scaler_feature_info()
    return jsonify(info)

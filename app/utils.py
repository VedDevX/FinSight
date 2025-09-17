# app/utils.py
import os
import joblib
import numpy as np
import pandas as pd

_BASE_DIR = os.path.dirname(__file__)
_MODELS_DIR = os.path.join(_BASE_DIR, "models")
_CACHE = {}

def _find_model_file(base_name):
    exts = [".pkl", ".pki", ".joblib", ".sav"]
    for ext in exts:
        p = os.path.join(_MODELS_DIR, base_name + ext)
        if os.path.exists(p):
            return p
    # exact filename fallback
    p_exact = os.path.join(_MODELS_DIR, base_name)
    if os.path.exists(p_exact):
        return p_exact
    raise FileNotFoundError(f"Model file not found for base name '{base_name}'. Tried extensions {exts} and exact name '{base_name}' in {_MODELS_DIR}")

def _load_model(base_name):
    if base_name in _CACHE:
        return _CACHE[base_name]
    path = _find_model_file(base_name)
    model = joblib.load(path)
    _CACHE[base_name] = model
    return model

# -----------------------
# Loan Approval (RandomForest)
# -----------------------
def load_approval_model():
    try:
        return _load_model("loan_approval_best_rf")
    except FileNotFoundError:
        return None

def preprocess_approval(form_data):
    # same exact features & engineered columns used during training
    def conv(inp, mapping, default=-1):
        if inp is None:
            return default
        return mapping.get(inp, default)

    def safe_float(v):
        try:
            return float(v)
        except:
            return 0.0

    gender_input = form_data.get("Gender")
    married_input = form_data.get("Married")
    dependents_input = form_data.get("Dependents")
    education_input = form_data.get("Education")
    self_employed_input = form_data.get("Self_Employed")
    credit_history_input = form_data.get("Credit_History")
    property_area_input = form_data.get("Property_Area")

    applicant_income = safe_float(form_data.get("ApplicantIncome"))
    coapplicant_income = safe_float(form_data.get("CoapplicantIncome"))
    loan_amount = safe_float(form_data.get("LoanAmount"))
    loan_amount_term = safe_float(form_data.get("Loan_Amount_Term"))

    gender = conv(gender_input, {"Male": 0, "Female": 1}, default=-1)
    married = conv(married_input, {"Yes": 1, "No": 0}, default=-1)
    if dependents_input is None:
        dependents = 0
    else:
        s = str(dependents_input).strip()
        dependents = 3 if s == "3+" else (int(s) if s.isdigit() else 0)

    education = conv(education_input, {"Graduate": 1, "Not Graduate": 0}, default=0)
    self_employed = conv(self_employed_input, {"Yes": 1, "No": 0}, default=0)
    credit_history = conv(credit_history_input, {"Yes": 1, "No": 0}, default=0)
    property_area = conv(property_area_input, {"Rural": 0, "Semiurban": 1, "Urban": 2}, default=0)

    total_income = applicant_income + coapplicant_income
    applicant_income_log = np.log(applicant_income + 1)
    loan_amount_log = np.log(loan_amount + 1)
    loan_term_log = np.log(loan_amount_term + 1)
    total_income_log = np.log(total_income + 1)

    df = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "TotalApplicantIncome": total_income,
        "ApplicantIncomeLog": applicant_income_log,
        "LoanAmountLog": loan_amount_log,
        "Loan_Amount_Term_Log": loan_term_log,
        "TotalApplicantIncomeLog": total_income_log
    }])
    return df

def predict_approval(form_data):
    model = load_approval_model()
    if model is None:
        return {"ok": False, "error": "Approval model not found. Place 'loan_approval_best_rf' file in app/models/ (supported ext: .pkl/.pki/.joblib/.sav)."}
    try:
        X = preprocess_approval(form_data)
        pred = model.predict(X)[0]
        out = {"ok": True, "label": int(pred)}
        if pred == 1:
            out["text"] = "Approved ✅"
            dependents = int(X.loc[0, "Dependents"])
            requested_loan_amount = float(form_data.get("LoanAmount") or 0.0)
            adjustment_factor = max(0.7, 1 - (0.1 * dependents))
            out["eligible_amount"] = round(requested_loan_amount * adjustment_factor, 2)
        else:
            out["text"] = "Rejected ❌"
            out["eligible_amount"] = 0.0
        return out
    except Exception as e:
        return {"ok": False, "error": f"Approval prediction error: {e}"}

# -----------------------
# Loan Default (XGBoost + Scaler)
# -----------------------
def load_default_model():
    try:
        return _load_model("xgboost_model_tuned")
    except FileNotFoundError:
        return None

def load_default_scaler():
    try:
        return _load_model("scaler")
    except FileNotFoundError:
        return None

def preprocess_default(form_data):
    # Create raw DataFrame and engineered features exactly as in your notebook
    def to_float(v):
        try:
            if v is None or str(v).strip() == "":
                return np.nan
            return float(v)
        except:
            return np.nan
    def to_int(v):
        try:
            if v is None or str(v).strip() == "":
                return 0
            return int(float(v))
        except:
            return 0

    RUL = to_float(form_data.get("RevolvingUtilizationOfUnsecuredLines"))
    age = to_int(form_data.get("age"))
    n30_59 = to_int(form_data.get("NumberOfTime30-59DaysPastDueNotWorse"))
    DebtRatio = to_float(form_data.get("DebtRatio"))
    MonthlyIncome_raw = form_data.get("MonthlyIncome")
    if MonthlyIncome_raw is None or str(MonthlyIncome_raw).strip() == "":
        MonthlyIncome = np.nan
    else:
        MonthlyIncome = to_float(MonthlyIncome_raw)

    open_lines = to_int(form_data.get("NumberOfOpenCreditLinesAndLoans"))
    n90 = to_int(form_data.get("NumberOfTimes90DaysLate"))
    real_estate = to_int(form_data.get("NumberRealEstateLoansOrLines"))
    n60_89 = to_int(form_data.get("NumberOfTime60-89DaysPastDueNotWorse"))
    dependents = to_int(form_data.get("NumberOfDependents"))

    new_data = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": RUL,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": n30_59,
        "DebtRatio": DebtRatio,
        "MonthlyIncome": MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfTimes90DaysLate": n90,
        "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": n60_89,
        "NumberOfDependents": dependents
    }])

    new_data["MonthlyIncome_missing"] = new_data["MonthlyIncome"].isnull().astype(int)
    new_data["LogDebtRatio"] = np.log1p(new_data["DebtRatio"].replace([np.inf, -np.inf], 0).fillna(0))
    new_data["LogMonthlyIncome"] = np.log1p(new_data["MonthlyIncome"].fillna(0))
    new_data["TotalPastDue"] = (
        new_data["NumberOfTime30-59DaysPastDueNotWorse"] +
        new_data["NumberOfTime60-89DaysPastDueNotWorse"] +
        new_data["NumberOfTimes90DaysLate"]
    )
    new_data["IncomePerOpenLine"] = new_data["MonthlyIncome"].fillna(0) / (new_data["NumberOfOpenCreditLinesAndLoans"] + 1)
    new_data["HasPastDelinquency"] = (new_data["TotalPastDue"] > 0).astype(int)

    # A reasonable default order (matches your notebook). We will adapt if model/scaler expect different names.
    feature_names = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
        "MonthlyIncome_missing",
        "LogDebtRatio",
        "LogMonthlyIncome",
        "TotalPastDue",
        "IncomePerOpenLine",
        "HasPastDelinquency"
    ]

    # ensure all columns exist
    for c in feature_names:
        if c not in new_data.columns:
            new_data[c] = 0.0

    new_data_processed = new_data[feature_names].copy()
    return new_data_processed, feature_names

def get_model_scaler_feature_info():
    """
    Return dict with scaler.feature_names_in_ (if present) and model.feature_names (if present).
    Useful for debugging mismatches on the user's machine.
    """
    info = {"scaler_feature_names_in": None, "model_feature_names": None}
    model = load_default_model()
    scaler = load_default_scaler()

    if scaler is not None:
        scaler_names = getattr(scaler, "feature_names_in_", None)
        if scaler_names is not None:
            info["scaler_feature_names_in"] = list(scaler_names)
        else:
            info["scaler_feature_names_in"] = None

    if model is not None:
        # Try various ways to get booster feature names
        try:
            booster = model.get_booster()
            bnames = getattr(booster, "feature_names", None)
            info["model_feature_names"] = list(bnames) if bnames is not None else None
        except Exception:
            # fallback
            mnames = getattr(model, "feature_names", None)
            info["model_feature_names"] = list(mnames) if mnames is not None else None

    return info

def predict_default(form_data, threshold=0.3):
    model = load_default_model()
    scaler = load_default_scaler()
    if model is None:
        return {"ok": False, "error": "Default model not found. Place 'xgboost_model_tuned' in app/models/."}
    if scaler is None:
        return {"ok": False, "error": "Scaler not found. Place 'scaler' in app/models/."}

    try:
        X_proc, initial_feature_names = preprocess_default(form_data)

        # If scaler knows its input feature names, reorder to that first
        scaler_expected = getattr(scaler, "feature_names_in_", None)
        if scaler_expected is not None:
            scaler_expected = list(scaler_expected)
            missing_for_scaler = [c for c in scaler_expected if c not in X_proc.columns]
            extra_produced = [c for c in X_proc.columns if c not in scaler_expected]
            if missing_for_scaler:
                return {
                    "ok": False,
                    "error": "Scaler expects feature names that are missing in preprocessed data.",
                    "details": {
                        "scaler_expected": scaler_expected,
                        "preprocessed_columns": list(X_proc.columns),
                        "missing_for_scaler": missing_for_scaler,
                        "extra_produced": extra_produced
                    }
                }
            # reorder columns to scaler expectation
            X_for_scaler = X_proc[scaler_expected]
        else:
            # Use the preprocessing order if scaler doesn't store feature names
            X_for_scaler = X_proc.copy()

        # scale -> numpy array
        X_scaled = scaler.transform(X_for_scaler)

        # rebuild DataFrame with columns = X_for_scaler.columns (so we preserve the intended names)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_for_scaler.columns)

        # get model expected feature names (if available)
        model_expected = None
        try:
            booster = model.get_booster()
            model_expected = getattr(booster, "feature_names", None)
            if model_expected is not None:
                model_expected = list(model_expected)
        except Exception:
            model_expected = getattr(model, "feature_names", None)
            if model_expected is not None:
                model_expected = list(model_expected)

        if model_expected is not None:
            missing_for_model = [c for c in model_expected if c not in X_scaled_df.columns]
            extra_for_model = [c for c in X_scaled_df.columns if c not in model_expected]
            if missing_for_model:
                # helpful error showing exact mismatch
                return {
                    "ok": False,
                    "error": "Model expects feature names that are missing after scaling.",
                    "details": {
                        "model_expected": model_expected,
                        "scaled_columns": list(X_scaled_df.columns),
                        "missing_for_model": missing_for_model,
                        "extra_for_model": extra_for_model
                    }
                }
            # reorder to model expectation
            X_for_model = X_scaled_df[model_expected]
        else:
            X_for_model = X_scaled_df

        proba = model.predict_proba(X_for_model)[:, 1]
        pred = (proba >= threshold).astype(int)
        return {"ok": True, "probability": float(proba[0]), "prediction": int(pred[0])}

    except Exception as e:
        return {"ok": False, "error": f"Default prediction error: {e}"}

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

preprocessor = joblib.load("preprocessor_pipeline.pkl")
model = joblib.load("ensemble_voting_model.pkl")


class LoanApplication(BaseModel):

    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


@app.get("/")
def home():
    return {"message": "Loan Approval Model Running"}


@app.post("/predict")
def predict(data: LoanApplication):

    input_df = pd.DataFrame([{
        "Gender": data.Gender,
        "Married": data.Married,
        "Dependents": data.Dependents,
        "Education": data.Education,
        "Self_Employed": data.Self_Employed,
        "ApplicantIncome": data.ApplicantIncome,
        "CoapplicantIncome": data.CoapplicantIncome,
        "LoanAmount": data.LoanAmount,
        "Loan_Amount_Term": data.Loan_Amount_Term,
        "Credit_History": data.Credit_History,
        "Property_Area": data.Property_Area
    }])

    transformed = preprocessor.transform(input_df)

    prediction = model.predict(transformed)[0]

    probability = model.predict_proba(transformed)[0]

    return {
        "loan_approved": int(prediction),
        "rejected_probability": round(float(probability[0]), 4),
        "approved_probability": round(float(probability[1]), 4)
    }
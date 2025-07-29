from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

class Profile(BaseModel):
    skills: list[str]
    certifications: list[str]
    seniority: str

app = FastAPI()
model = joblib.load("models/salary_model/ensemble.joblib")
@app.post("/predict")
def predict(profile: Profile):
    # construye un DataFrame de una fila con las columnas de features
    row = {col: 0 for col in model.feature_names_in_}
    for s in profile.skills:       row[f"skill_{s}"] = 1
    for c in profile.certifications: row[f"certification_{c}"] = 1
    # seniority, length, num_entities si quieres
    df = pd.DataFrame([row])
    salary = model.predict(df)[0]
    return {"predicted_salary": salary}

"""Task 5 - Model serving with FastAPI.

Run:  uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["model"] = joblib.load("model.pkl")      # same pipeline used everywhere
    yield
    ml.clear()


app = FastAPI(title="Titanic Survival Prediction API", version="1.0",
              lifespan=lifespan)


class Passenger(BaseModel):
    Pclass: Literal[1, 2, 3] = Field(..., description="Ticket class (1, 2 or 3)")
    Sex: Literal["male", "female"]
    Age: Optional[float] = Field(None, ge=0, le=120)
    SibSp: int = Field(..., ge=0, le=10, description="Siblings/spouses aboard")
    Parch: int = Field(..., ge=0, le=10, description="Parents/children aboard")
    Fare: float = Field(..., ge=0, le=600)
    Embarked: Optional[Literal["C", "Q", "S"]] = None

    model_config = {"json_schema_extra": {"example": {
        "Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0,
        "Parch": 0, "Fare": 100.0, "Embarked": "S"}}}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in ml}


@app.post("/predict")
def predict(p: Passenger):
    df = pd.DataFrame([p.model_dump()])
    model = ml["model"]
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {"prediction": pred,
            "result": "Survived" if pred else "Did not survive",
            "survival_probability": round(prob, 4)}

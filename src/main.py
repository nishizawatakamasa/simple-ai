import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from sklearn.linear_model import LinearRegression

app = FastAPI()

model: LinearRegression = joblib.load('models/model.joblib')

class InputData(BaseModel):
    value: float

    @field_validator('value', mode='before')
    @classmethod
    def value_must_be_positive(cls, value):
        if value <= 0:
            raise ValueError('Value must be positive')
        return value

class PredictionResult(BaseModel):
    input: float
    prediction: float

@app.post('/predict/', response_model=PredictionResult)
def predict(data: InputData) -> PredictionResult:
    '''This endpoint takes a single value as input and returns the model's prediction.'''
    input_value = data.value
    prediction = model.predict(np.array([[input_value]]))[0]
    return PredictionResult(input=input_value, prediction=prediction)

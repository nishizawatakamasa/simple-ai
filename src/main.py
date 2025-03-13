import joblib
import numpy as np
from fastapi import FastAPI

app = FastAPI()

model = joblib.load('models/model.joblib')

@app.get('/predict/')
def predict(value: float):
    prediction = model.predict(np.array([[value]]))
    return {'input': value, 'prediction': prediction[0].item()}

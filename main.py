from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict

app = FastAPI()


class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict_endpoint(data: InputData):

    prediction = predict(data.features)
    return {"prediction": prediction}

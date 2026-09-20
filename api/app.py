from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import load_model

app = FastAPI(title="AI Operations Request Router")
model = load_model()


class RoutingRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: RoutingRequest):
    label = str(model.predict([request.text])[0])
    return {"predicted_topic": label}

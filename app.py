from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import torch
from model_training import BiLSTMClassifier

app = FastAPI(
    title="Market Direction Predictive MLOps API",
    description="Real-time Inference Endpoint & Feature Drift Monitor"
)

# Load lightweight PyTorch model for inference
MODEL_PATH = "bilstm_demo.pt"
input_dim = 4
model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=64)
model.eval()

class SequenceInput(BaseModel):
    # Expecting 10 time-steps x 4 features ['MACD', 'RSI_14', 'Volatility_14', 'Log_Return']
    sequence: list[list[float]] = Field(..., example=[[0.1, 55.0, 0.02, 0.001] for _ in range(10)])

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "MLOps Predictor"}

@app.post("/predict")
def predict_direction(payload: SequenceInput):
    seq = np.array(payload.sequence)
    if seq.shape != (10, 4):
        raise HTTPException(status_code=400, detail=f"Expected shape (10, 4), got {seq.shape}")

    tensor_input = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probability = model(tensor_input).item()

    prediction = "UP" if probability > 0.5 else "DOWN"
    return {
        "direction_prediction": prediction,
        "upward_probability": round(probability, 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pathlib import Path
import logging

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

block_dir = Path("../data/blocks")
model_name = "DeepSeek-R1-Distill-Qwen-1.5B-ONNX"

@app.get("/model")
async def get_model():
    model_path = block_dir / model_name / f"{model_name}_split.onnx"
    with open(model_path, "rb") as f:
        return StreamingResponse(iter([f.read()]), media_type="application/octet-stream")

@app.get("/weights/{filename}")
async def get_weights(filename: str):
    weight_path = block_dir / model_name / filename
    if weight_path.exists():
        with open(weight_path, "rb") as f:
            return StreamingResponse(iter([f.read()]), media_type="application/octet-stream")
    return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.model.cnn_model import predict_pipeline
from app.model.cnn_model import __version__ as model_version

app = FastAPI()

#
@app.get("/")
async def root():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    is_fire = await predict_pipeline(file)
    return JSONResponse(content={"is_fire": bool(is_fire)})
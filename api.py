
from fastapi import FastAPI
from transformers import AutoModel

app = FastAPI()

model = None

@app.on_event("startup")
async def load_model():
    global model
    model = AutoModel.from_pretrained("JeffreyXiang/TRELLIS-image-large")

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.get("/model-loaded")
async def is_model_loaded():
    return {"model_loaded": model is not None}


#for local development
from fastapi import FastAPI, UploadFile, File
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import uvicorn
import requests
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()
endpoint="http://localhost:8501/v1/models/potato_model:predict"


#from fastapi import FastAPI



# Allow requests from your React app
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#MODEL = tf.keras.models.load_model("../models/1")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/1")

MODEL = tf.keras.models.load_model(MODEL_PATH)
class_names = ["Early_blight", "Late_blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello am alive chris"

#Read file and convert into numpy array
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))  # match your model input size
    image = np.array(image)
    return image

#performs predictions

@app.post("/predict")

async def prediction(file : UploadFile=File(...)):

    image= read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    json_data = {
    "instances": img_batch.tolist()
}
    response = requests.post(endpoint, json=json_data)
    result = np.array(response.json()["predictions"][0])
    
    predicted_class = class_names[np.argmax(result)]
    confidence = np.max(result)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
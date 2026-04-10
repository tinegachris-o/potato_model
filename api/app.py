#for API development
from fastapi import FastAPI, UploadFile, File
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# IMPORTANT: Load model directly from folder 4 (no local host needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assumes folder '4' is in the same directory as this script
MODEL_PATH = os.path.join(BASE_DIR, "4") 
MODEL = tf.keras.models.load_model(MODEL_PATH)

class_names = ["Early_blight", "Late_blight", "Healthy"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allowing all for the cloud test
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return "Hello! The Cloud Engine is alive."

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).resize((256, 256))
    return np.array(image)

@app.post("/predict")
async def prediction(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    # Predict directly using the loaded MODEL
    predictions = MODEL.predict(img_batch)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    # Cloud Run expects port 8080
    port = int(os.environ.get("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
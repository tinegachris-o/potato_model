# 🥔 Potato Disease Classifier API

A high-performance deep learning API that classifies potato leaf diseases using **TensorFlow** and **FastAPI**. This project is designed to help farmers identify **Early Blight**, **Late Blight**, and **Healthy** leaves in real-time.
![Potato Disease Classifier Interface](img.png)
## 🚀 Features
* **Model:** Convolutional Neural Network (CNN) trained on the PlantVillage dataset.
* **API:** Built with FastAPI for high-performance, asynchronous image processing.
* **Containerized:** Ready for deployment via Docker.
* **Accuracy:** ~98% (or insert your specific accuracy here).

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Framework:** FastAPI / Uvicorn
* **ML Library:** TensorFlow / Keras
* **Image Processing:** Pillow (PIL), NumPy

## 📁 Project Structure
```text
.
├── api/
│   ├── app.py           # FastAPI entry point
│   ├── requirements.txt # Project dependencies
│   └── 4/               # The exported TensorFlow model
├── models/              # Training scripts and model history
└── README.md
"""

serving my model using 
sudo docker run -p 8501:8501 \
  --name potato_container \
  --mount type=bind,source=$(pwd)/potato_model,target=/models/potato_model \
  -e MODEL_NAME=potato_model \
  -t tensorflow/serving

"""
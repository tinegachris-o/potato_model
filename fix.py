import tensorflow as tf

import os

model_dir = "models/3"

try:
    # 1. Reconstruct the model architecture
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = f.read()
    
    model = tf.keras.models.model_from_json(config)
    
    # 2. Load the weights
    model.load_weights(os.path.join(model_dir, "model.weights.h5"))
    
    # 3. Use .export() for TF Serving (This creates the saved_model.pb)
    model.export("models/4")
    print("✅ Success! Folder 'models/4' now contains the saved_model.pb file.")

except Exception as e:
    print(f"❌ Error: {e}")
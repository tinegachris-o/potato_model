import tensorflow as tf
import os
import shutil

h5_path = "/home/tinega/Documents/projects/Potato/models/potato_model.h5"
export_dir = "/home/tinega/Documents/projects/Potato/models/potato_model/5"

# Clean up the directory if it exists to avoid conflicts
if os.path.exists(export_dir):
    shutil.rmtree(export_dir)

print("Loading .h5 model...")
model = tf.keras.models.load_model(h5_path)

print("Exporting to SavedModel format for TF Serving...")
# In Keras 3, this is the most reliable way to get a SavedModel folder
model.export(export_dir) 

print(f"Successfully exported to {export_dir}")
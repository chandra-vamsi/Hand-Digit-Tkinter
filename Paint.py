import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# Load the trained model
model = keras.models.load_model("model.h5")

# Load the painted image and preprocess it
try:
    img = Image.open("image.png").convert("L")  # convert to grayscale
    img = img.resize((28, 28))  # resize to 28x28
    img_array = np.array(img)  # convert to numpy array
    img_array = img_array.reshape((1, 28 * 28))  # reshape to match the input shape of the model
    img_array = img_array.astype("float32") / 255  # normalize the pixel values
except Exception as e:
    print(f"Error loading or preprocessing image: {e}")
    exit()

# Predict the number using the model
try:
    pred = model.predict(img_array)
    number = np.argmax(pred)
except Exception as e:
    print(f"Error predicting number: {e}")
    exit()

print(f"The painted number is: {number}")
print(f"Prediction probabilities: {pred}")

import tkinter as tk
import tkinter.messagebox as messagebox
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model = load_model("model.h5")

# Define the size of the drawing canvas
canvas_width = 280
canvas_height = 280

# Create a new Tkinter window
window = tk.Tk()
window.title("Handwriting Recognition")

# Create a canvas for drawing
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

# Create a button for recognition
button = tk.Button(window, text="Recognize", command=lambda: recognize(canvas))
button.pack()

# Create a PIL Image object for drawing
img = Image.new("L", (canvas_width, canvas_height), color="white")

# Create a PIL ImageDraw object for drawing
draw = ImageDraw.Draw(img)

# Define the brush size and color
brush_size = 20
brush_color = "black"

# Define a function for drawing on the canvas
def paint(event):
    x1, y1 = (event.x - brush_size), (event.y - brush_size)
    x2, y2 = (event.x + brush_size), (event.y + brush_size)
    canvas.create_oval(x1, y1, x2, y2, fill=brush_color, outline=brush_color)
    draw.ellipse([x1, y1, x2, y2], fill=brush_color, outline=brush_color)

# Bind the paint function to the canvas
canvas.bind("<B1-Motion>", paint)

# Define a function for recognizing the drawn digit
def recognize(canvas):
    # Resize the drawn image to the expected size and preprocess it
    img_resized = img.resize((28, 28))
    img_array = np.array(img_resized)
    img_array = img_array.astype("float32") / 255
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    # Predict the digit using the model
    pred = model.predict(img_array)
    digit = np.argmax(pred)

    # Display the predicted digit in a messagebox
    tk.messagebox.showinfo("Recognition", f"The drawn digit is: {digit}")

# Start the Tkinter event loop
window.mainloop()

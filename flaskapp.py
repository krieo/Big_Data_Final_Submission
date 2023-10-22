import os

from flask import Flask, request, render_template
import cv2
import numpy as np
from keras.models import load_model

app = Flask("This is a sample")
# Define the folder where you want to save your models
model_folder = "models"
model_filename = "my_model.h5"
model_path = os.path.join(model_folder, model_filename)

model = load_model(model_path)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        resized_image = cv2.resize(img, (256, 256))
        normalized_image = resized_image / 255.0
        expanded_image = np.expand_dims(normalized_image, axis=0)

        yhat = model.predict(expanded_image)
        predicted_class = np.argmax(yhat)

        class_names = ["Anna", "Bana", "Cana", "Dana", "Eana", "Fana"]
        predicted_label = class_names[predicted_class]

        return f"Predicted Class: {predicted_label}"
    else:
        return "No file uploaded."


if __name__ == '__main__':
    app.run(debug=True)

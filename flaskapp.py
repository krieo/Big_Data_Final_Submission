import os

from flask import Flask, request, render_template
import cv2
import numpy as np
from keras.models import load_model

app = Flask("Mosquito detection")
# This gets the model that was created in the model folder
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

        predictions = model.predict(expanded_image)
        predicted_class = np.argmax(predictions)

        class_names = ["Aegypti", "Albopictus", "Anopheles", "Culex", "Culiseta", "Japonicus_Korecius"]
        predicted_label = class_names[predicted_class]
        probability = 0
        for i, prediction in enumerate(predictions):
            predicted_class = class_names[np.argmax(prediction)]
            probability = max(prediction)
            print(f"Predicted Class - {predicted_class}, Probability - {probability:.2f}")

        return f"Predicted Class - {predicted_class}, Probability - {probability:.2f}"#return f"Predicted Class: {predicted_label}"
    else:
        return "No file uploaded."


if __name__ == '__main__':
    app.run(debug=True)

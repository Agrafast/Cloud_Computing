import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
import h5py
from tensorflow import keras
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify

app = Flask(__name__)

models = {
    'rice': {
        'model_path': 'cnn-model-rice.h5',
        'class_names': ["bacterial_leaf_blight", "brown_spot", "healthy", "leaf_blast", "leaf_scald", "narrow_brown_spot"]
    },
    'maize': {
        'model_path': 'cnn-model-corn.h5',
        'class_names': ['blight', 'common_rust', 'gray_leaf_spot', 'healthy']
    },
    'potato': {
        'model_path': 'cnn-model-potato.h5',
        'class_names': ['early_blight', 'healthy', 'late_blight']
    }
}

def load_model(model_path):
    if not os.path.isfile(model_path):
        print(f"Error: File '{model_path}' not found.")
        # Handle the error appropriately, such as logging or raising an exception.
    else:
        return keras.models.load_model(model_path)

def transform_image(pillow_image):
    data = np.asarray(pillow_image)
    data = data / 255.0
    data = tf.image.resize(data, [224, 224])
    data = np.expand_dims(data, axis=0)
    return data

def predict(model, x):
    predictions = model(x)
    predictions = tf.nn.softmax(predictions)
    pred0 = predictions[0]
    label0 = np.argmax(pred0)
    return label0

@app.route("/")
def index():
    return "OK"

@app.route("/index/<model_choice>", methods=["POST"])
def predict_image(model_choice):
    if model_choice not in models:
        return jsonify({'error': 'Invalid model choice'})

    model_info = models[model_choice]
    model_path = model_info['model_path']
    class_names = model_info['class_names']

    model = load_model(model_path)

    if model is None:
        return jsonify({'error': 'Failed to load model'})

    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "No file"})

        try:
            image_bytes = file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes))
            tensor = transform_image(pillow_img)
            prediction = predict(model, tensor)
            label = class_names[prediction]
            data = {"prediction": label}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    # return "OK"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)

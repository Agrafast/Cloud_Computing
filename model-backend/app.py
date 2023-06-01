from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predictImage():
    imagefile = request.files['image']
    img_path = "./images/" + imagefile.filename
    imagefile.save(img_path)

    image = load_img(img_path, target_size=(224, 224))
    image = np.array(image)/255
    image = np.expand_dims(image, axis=0)

    model_choice = request.form.get('model')

    if model_choice == 'rice':
        cnn_model = 'saved_model/cnn-model-rice.h5'
        class_names = ["Bacterial leaf blight", "Brown spot", "Healthy", "Leaf blast", "Leaf scald", "Narrow brown spot"]
    elif model_choice == 'potato':
        cnn_model = 'saved_model/cnn-model-potato.h5'
        class_names = ["Early Blight", "Healthy", "Late Blight"]
    elif model_choice == 'corn':
        cnn_model = 'saved_model/cnn-model-corn.h5'
        class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    else:
        return ("Maaf anda belum memilih tanaman")

    model = tf.keras.models.load_model(cnn_model)
    probs = model.predict(image)
    high = np.argmax(probs)

    result = class_names[high]

    # get accuracy for each class
    class_probs = {}
    for i in range(len(class_names)):
        class_probs[class_names[i]] = round(float(probs[0][i]*100), 2)

    return render_template('index.html', result=result, class_probs=class_probs)
    # return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(port=3000, debug=True)
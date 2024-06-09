import os
import string
import random
import json
import requests
import numpy as np
import tensorflow as tf

from flask import Flask, request, redirect, url_for, render_template
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

"""
Constants
"""
MODEL_URI = 'http://localhost:8502/v1/models/cifar10:predict'
OUTPUT_DIR = 'static'
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
SIZE = 32

"""
Utility functions
"""
def generate_filename():
    return ''.join(random.choices(string.ascii_lowercase, k=20)) + '.jpg'

def get_prediction(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(SIZE, SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    data = json.dumps({'instances': image.tolist() })
    response = requests.post(MODEL_URI, data=data.encode())
    result = json.loads(response.text)
    prediction = result['predictions'][0]
    class_index = np.argmax(prediction)
    class_name = CLASSES[class_index]
    return class_name

"""
Routes
"""
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            if uploaded_file.filename[-3:] in ['jpg', 'png']:
                image_path = os.path.join(OUTPUT_DIR, generate_filename())
                uploaded_file.save(image_path)
                class_name = get_prediction(image_path)
                result = {
                    'class_name': class_name,
                    'path_to_image': image_path,
                    'size': SIZE
                }
                return render_template('show.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

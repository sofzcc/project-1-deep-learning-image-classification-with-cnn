import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash
from PIL import Image
import numpy as np
import json
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SECRET_KEY'] = 'your_secret_key'  # Add a secret key for flash messages

# CIFAR-10 class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image):
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    return image

def get_prediction(image):
    # Make prediction request to your model server
    url = 'http://localhost:8501/v1/models/image_classification_model:predict'
    data = json.dumps({"instances": image[None, ...].tolist()})
    headers = {"content-type": "application/json"}
    response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(response.text)['predictions']
    return predictions

@app.route("/")
def home():
    return render_template("index.html", class_names=class_names)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image
        image = Image.open(filepath)
        preprocessed_image = preprocess_image(image)
        
        # Get prediction
        predictions = get_prediction(preprocessed_image)
        
        predicted_class = class_names[np.argmax(predictions)]
        
        return render_template('result.html', filename=filename, predicted_class=predicted_class)
    
    return redirect(request.url)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000)

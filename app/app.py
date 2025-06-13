import os
from flask import Flask, request, render_template, url_for, send_from_directory
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the pre-trained model
model = keras.models.load_model('app/data/cnn_citra64.h5')

# List of class labels (replace with your actual class labels)
class_labels = ['Cross Training', 'Road Running', 'Trail Running']  # Example class labels

# Function to preprocess the input image using OpenCV
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))  # Resize the image to match the input size of the model
    img = img / 255.0  # Normalize the image
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def classify_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    confidence_score = predictions[0][predicted_index]
    return predicted_class, confidence_score

@app.route('/', methods=['GET', 'POST'])
def index():
    # image_path = None
    if request.method == 'POST':
        if 'resume' not in request.files:
            return 'No file part'
        file = request.files['resume']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = file.filename

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            predicted_class, confidence_score = classify_image(file_path)
            # Generate the URL for the uploaded image
            image_url = url_for('uploaded_file', filename=file.filename)
            return render_template('index.html', prediction=predicted_class, confidence=confidence_score, image_path=image_url)
    return render_template('index.html', prediction=None, confidence=None, image_path=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
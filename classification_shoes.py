import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained model
model = keras.models.load_model('app/data/cnn_citra64.h5')
print(model.input_shape)
print(model.summary())

# List of class labels (replace with your actual class labels)
class_labels = ['Cross Training', 'Road Running', 'Trail Running']  # Example class labels

# Function to preprocess the input image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Resize the image to match the input size of the model
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def classify_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    confidence_score = predictions[0][predicted_index]
    return predicted_class, confidence_score

# Example usage
image_path = 'app/data/rr_s1527.jpeg'  # Replace with the path to your image
predicted_class, confidence_score = classify_image(image_path)
print(f'Predicted class: {predicted_class} with confidence score: {confidence_score:.2f}')

# Display the image and prediction
img = Image.open(image_path)
plt.imshow(img)
plt.title(f'Prediction: {predicted_class} ({confidence_score:.2f})')
plt.show()


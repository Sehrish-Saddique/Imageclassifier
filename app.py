from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import os

app = Flask(__name__)

# Load pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to predict image classes
def predict_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))  # Load image with target size
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model input
    img_array = preprocess_input(img_array)  # Preprocess the input image
    predictions = model.predict(img_array)  # Make predictions
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Decode predictions
    return [{'label': label, 'confidence': float(confidence*100)} for _, label, confidence in decoded_predictions]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        predictions = predict_image(file_path)
        os.remove(file_path)  # Remove the uploaded file after prediction
        return jsonify({'predictions': predictions})

    return jsonify({'error': 'Something went wrong'})

if __name__ == '__main__':
    app.run(debug=True)

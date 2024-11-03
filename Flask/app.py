from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import time

# Initialize Flask app
app = Flask(__name__)

# Load the trained models
model_serial = load_model('C:/Users/DELL/Flask/papaya_disease_prediction_serial.h5')
model_parallel = load_model('C:/Users/DELL/Flask/papaya_disease_prediction_parallel.h5')
class_names = ["Healthy", "Anthracnose", "Phytophthora blight", "Brown spot", "Black spot", "Other diseases"]

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Process the uploaded image
    if file:
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        img = load_img(filepath, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Measure prediction time for the serial model
        start_time = time.time()
        predictions_serial = model_serial.predict(img_array)
        predicted_class_serial = class_names[np.argmax(predictions_serial)]
        time_serial = time.time() - start_time

        # Measure prediction time for the parallel model
        start_time = time.time()
        predictions_parallel = model_parallel.predict(img_array)
        predicted_class_parallel = class_names[np.argmax(predictions_parallel)]
        time_parallel = time.time() - start_time

        # Check if parallel model is faster
        is_parallel_faster = time_parallel < time_serial

        return render_template('result.html', 
                               prediction_serial=predicted_class_serial, 
                               prediction_parallel=predicted_class_parallel, 
                               time_serial=time_serial, 
                               time_parallel=time_parallel, 
                               is_parallel_faster=is_parallel_faster, 
                               img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)

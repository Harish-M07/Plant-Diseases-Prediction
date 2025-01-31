import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Path to the saved model
model = load_model('model.keras')
print("Model loaded successfully!")

# Labels for prediction
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Folder to store uploaded images temporarily
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to predict the disease
def get_prediction(image_path):
    img = load_img(image_path, target_size=(225, 225))
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)[0]
    predicted_label = labels[np.argmax(predictions)]
    return predicted_label

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file uploaded!"
        file = request.files['file']

        if file.filename == '':
            return "No selected file!"

        if file:
            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get prediction
            result = get_prediction(file_path)
            return render_template('index.html', prediction=result)

    return render_template('index.html')

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)

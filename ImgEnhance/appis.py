from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename
from skimage import exposure

app = Flask(__name__)

# Directory to save uploaded and processed images
UPLOAD_FOLDER = 'static/uploads/'
ENHANCED_FOLDER = 'static/enhanced/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ENHANCED_FOLDER'] = ENHANCED_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(ENHANCED_FOLDER):
    os.makedirs(ENHANCED_FOLDER)

# Function to apply image enhancement techniques
def apply_image_enhancements(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Histogram Equalization
    hist_eq_image = cv2.equalizeHist(image)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image)

    # Apply Unsharp Masking (to enhance edges)
    gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
    unsharp_image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    # Noise Reduction (using Non-Local Means Denoising)
    denoised_image = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)

    # Edge Detection (using Canny Edge Detection)
    edges_image = cv2.Canny(image, 100, 200)

    # Save all enhanced images
    filenames = {}
    filenames['hist_eq_image'] = save_image(hist_eq_image, 'hist_eq_image.png')
    filenames['clahe_image'] = save_image(clahe_image, 'clahe_image.png')
    filenames['unsharp_image'] = save_image(unsharp_image, 'unsharp_image.png')
    filenames['denoised_image'] = save_image(denoised_image, 'denoised_image.png')
    filenames['edges_image'] = save_image(edges_image, 'edges_image.png')

    return filenames

# Helper function to save the image and return the path
def save_image(image_array, filename):
    file_path = os.path.join(app.config['ENHANCED_FOLDER'], filename)
    cv2.imwrite(file_path, image_array)
    return file_path

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image upload and enhancement
@app.route('/upload', methods=['GET','POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded image
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Apply image enhancement techniques
        enhanced_images = apply_image_enhancements(file_path)

        # Render the result page
        return render_template('result.html', original_image=url_for('static', filename=f'uploads/{filename}'), enhanced_images=enhanced_images)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

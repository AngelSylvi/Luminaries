import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# Initialize Flask app
app = Flask(__name__)

# Load the saved model from the Jupyter notebook
model_03 = load_model('MedImgDiag/brain_mri_detection.h5')  # Load the full model with architecture and weights

model_03 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(240, 240, 3)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])


# Function to map predicted class index to tumor type
def get_className(classNo):
    if classNo == 0:
        return "No Tumor"
    elif classNo == 1:
        return "Glioma Tumor"
    elif classNo == 2:
        return "Meningioma Tumor"
    elif classNo == 3:
        return "Pituitary Tumor"


def getResult(img_path):
    # Read the image from the given path
    image = cv2.imread(img_path)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Resize to the required input size (240x240)
    image = image.resize((240, 240))

    # Convert to array and preprocess
    image = np.array(image).astype('float32') / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Run the image through the model
    result = model_03.predict(image)

    # Get the predicted class
    class_index = np.argmax(result, axis=1)
    return class_index[0]


# Define the home route to display the upload form
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Define the route to handle the image upload and make the prediction
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('image')  # Get the uploaded file
        if f and f.filename:  # Check if a file was uploaded and has a filename
            # Create a secure file path to store the uploaded image
            basepath = os.path.dirname(__file__)
            upload_folder = os.path.join(basepath, 'uploads')
            
            # Create the uploads directory if it doesn't exist
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
                
            file_path = os.path.join(upload_folder, secure_filename(f.filename))
            print(f"File will be saved to: {file_path}")  # Debugging line
            try:
                f.save(file_path)  # Save the uploaded image to the server
            except Exception as e:
                return jsonify(error=f"Error saving file: {str(e)}"), 500

            # Get the prediction result for the uploaded image
            predicted_class = getResult(file_path)
        
        
        # Map the predicted class index to its corresponding tumor category
        result = get_className(predicted_class)
        
        # Return the result to be displayed in the UI
        return result
    return None

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

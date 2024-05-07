from flask import Flask, request, jsonify, send_from_directory, url_for
import base64
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import datetime

# Load the model from files
with open("model4.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights("model_weights4.h5")

# Define class names and their corresponding image paths
class_images = {
    'Am': 'Am.png',
    'Db': 'Db.png',
    'Em': 'Em.png',
    'F': 'F.png',
    'G#m': 'G#m.png'
}

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'predicted_images')

# Make sure the static images directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route('/')
def serve_html():
    return send_from_directory('static', 'index.html')


@app.route('/save_and_predict', methods=['POST'])
def save_and_predict():
    data = request.get_json()
    img_data = data['image'].split(',')[1]  # Extract the base64 image data
    decoded_img = base64.b64decode(img_data)
    image = Image.open(io.BytesIO(decoded_img)).convert("RGB")

    # Save the uploaded image (optional)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], f'image_{timestamp}.jpg')
    image.save(img_path, "JPEG")

    # Prepare the image for prediction
    image = image.resize((224, 224))  # Adjust size to your model's input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)

    # Make the prediction
    predictions = loaded_model.predict(image)
    predicted_class = np.argmax(predictions)
    predicted_class_name = list(class_images.keys())[predicted_class]

    # Get the path to the predicted image
    predicted_image = class_images[predicted_class_name]
    image_url = url_for('static', filename=predicted_image)

    return jsonify({'class': predicted_class_name, 'image_url': image_url})


if __name__ == '__main__':
    app.run(debug=True)

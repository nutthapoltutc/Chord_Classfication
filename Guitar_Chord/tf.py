import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

with open("model3.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

loaded_model.load_weights("model_weights3.h5")

image_path = "img_2.png"
image = Image.open(image_path)
image = image.resize((150, 150))
image = image.convert("RGB")
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

predictions = loaded_model.predict(image)
predicted_class = np.argmax(predictions)

class_names = ['Am', 'C#', 'C#m', 'F#', 'F#m']  # Define your class names here
predicted_class_name = class_names[predicted_class]

print("Predicted class:", predicted_class_name)

image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.title("Predicted class: {}".format(predicted_class_name))
plt.show()

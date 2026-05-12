import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_NAMES = [
    'Black Scurf',
    'Blackleg',
    'Common Scab',
    'Dry Rot',
    'Pink Rot',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy'
]


class PotatoCNNClassifier:

    def __init__(self):

        self.model = tf.keras.models.load_model(
            "src/cnn/models/potato_efficientnet.keras"
        )

    def preprocess_image(self, image):

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")
        image = image.resize((224, 224))

        img_array = np.array(image)

        img_array = tf.keras.applications.efficientnet.preprocess_input(
            img_array
        )

        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image):

        processed = self.preprocess_image(image)

        predictions = self.model.predict(processed, verbose=0)[0]

        predicted_idx = np.argmax(predictions)

        confidence = float(predictions[predicted_idx]) * 100

        prediction = CLASS_NAMES[predicted_idx]

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "all_scores": {
                CLASS_NAMES[i]: round(float(score) * 100, 2)
                for i, score in enumerate(predictions)
            }
        }
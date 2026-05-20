import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


IMG_SIZE = (224, 224)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a chest X-Ray image for MobileNetV2 binary classification.

    Returns a batch tensor with shape (1, 224, 224, 3).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(IMG_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    batch = np.expand_dims(image_array, axis=0)

    return batch


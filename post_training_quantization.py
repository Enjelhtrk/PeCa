from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import numpy as np
import os

def representative_dataset():
    Batch_size = 32

    directory = "../PECA/Dataset"
    categories = ["scabies", "sehat"]

    data = []
    labels = []

    for category in categories:
        path = os.path.join(directory, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)

            data.append(image)

    data = np.array(data, dtype="float32")

    dataset = tf.data.Dataset.from_tensor_slices(data).batch(1)

    for input_value in dataset.take(Batch_size):
        yield [input_value]

model = load_model("skin_disease_detector.model")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open('modelan.tflite', 'wb') as f:
    f.write(tflite_quant_model)
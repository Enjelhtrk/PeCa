from keras.models import load_model
import tensorflow as tf

model = load_model("skin_disease_detector.model")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_quant_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_quant_model)
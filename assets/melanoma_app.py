# imports
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout
import os
from flask import Flask, request, render_template, jsonify


# load model
json_string='{"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": [null, 224, 224, 3]}, "keras_version": "2.2.4-tf", "backend": "tensorflow"}'

model = model_from_json(json_string)

# app
app = Flask(__name__)

# routes
@app.route('/')

def results(img_path, show=True):



    if __name__ == "__main__":
        app.run(host='127.0.0.1', port=5000)

        img = image.load_img(img_path, target_size=(224, 224))
        img_tensor = image.img_to_array(img)                    # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

        # image path
        img_path = img_path

        # load a single image
        new_image = load_image(img_path)

        # check prediction
        pred = model.predict(new_image)

    if pred[0][0] >= np.round(0.510, 2):

        print('The Results are in:')
        print(pred[0][0])

    if pred[0][1] >= np.round(0.510, 2):

        print('The Results are in:')
        print(pred[0][1])

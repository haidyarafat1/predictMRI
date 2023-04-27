# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 03:28:54 2023

@author: DELL
"""

from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

def preprossing(image):
    image=Image.open(image)
    image = image.resize((128, 128))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 150, 150, 3)
    return image_arr

classes = ['Mild_Demented' ,'Moderate_Demented', 'Non_Demented' ,'Very_Mild_Demented' ]
model=load_model("C:/Users/DELL/Downloads/My_saved_Model.h5")

@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)
        print("Model predicting ...")
        result = model.predict(image_arr)
        print("Model predicted")
        ind = np.argmax(result)
        prediction = classes[ind]
        print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})

if __name__ == '__main__':
    app.run(debug=True)

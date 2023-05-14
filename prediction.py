# -*- coding: utf-8 -*-
"""
Created on Sun May 14 01:23:23 2023

@author: Ajay
"""

from PIL import Image
import numpy as np
from PIL import Image, ImageOps
import keras
import cv2
import streamlit as st
def mri_classification(img, weights_file):
    cnn_model = keras.models.load_model(weights_file)
    """
    
    flat_data=[]
    img=np.array(img)
    img_resized=cv2.resize(img,(128,128))
    img_resized=img_resized/255
    flat_data.append(img_resized)
    flat_data=np.array(flat_data)
    flat_data = np.reshape(flat_data, (-1, 128, 128, 3))
    
    input_prediction = cnn_model.predict(flat_data)
    
    input_pred_label = np.argmax(input_prediction)
    """
    
    img_array = np.array(img)  # Convert image to NumPy array
    img_resized = cv2.resize(img_array, (128, 128))
    img_resized = img_resized / 255.0
    
    # Check if the image is grayscale (single channel)
    if len(img_resized.shape) == 2:
        # Convert grayscale image to RGB by repeating the single channel 3 times
        img_resized = np.repeat(img_resized[..., np.newaxis], 3, axis=-1)
    
    flat_data = np.array([img_resized])
    
    input_prediction = cnn_model.predict(flat_data)
    input_pred_label = np.argmax(input_prediction)
    
    
    
    
    return input_pred_label
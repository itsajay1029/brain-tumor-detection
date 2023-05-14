# -*- coding: utf-8 -*-
"""
Created on Sun May 14 00:55:57 2023

@author: Ajay
"""

import streamlit as st
from PIL import Image 
from prediction import mri_classification
import os
st.title("Brain Tumor or Healthy Brain")
st.header("Brain Tumor MRI Classifier")
st.text("Upload a brain MRI Image for image classification as tumor or Healthy Brain")
     
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("")
    app_folder = os.path.dirname(os.path.abspath(__file__))
    mri_classifier_path = os.path.join(app_folder, "mri_classifier.h5")
    label = mri_classification(image, mri_classifier_path)
    
    if label == 0:
       st.write("The MRI scan shows an healthy brain")
    else:
       st.write("The MRI scan detects a brain tumor")
   
        
        
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 00:20:48 2021

@author: Mohammed Shreef
"""


#from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
#import flasgger
import streamlit as st
#from flasgger import Swagger

# app=Flask(__name__)
# Swagger(app)
pickle_in=open('xgb_classifier_2.pkl','rb')
classifier=pickle.load(pickle_in)

# pickle_in_vect=open('std_scaler.pkl','rb')
# transformer=pickle.load(pickle_in_vect)

#@app.route('/')
def welcome():
    return 'welcome sharif'

#@app.route('/predict',methods=["Get"])



def thermal_comfort_prediction(clo_insulation,metabolic_rate,air_temparature,relative_humidity,air_velocity):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: air_temparature
        in: query
        type: number
        required: true
      - name: radiant_temparature
        in: query
        type: number
        required: true
      - name: relative_humidity
        in: query
        type: number
        required: true
      - name: air_velocity
        in: query
        type: number
        required: true
      - name: clo_insulation
        in: query
        type: number
        required: true
      - name: metabolic_rate
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    vect= np.array([[clo_insulation,metabolic_rate,air_temparature,relative_humidity,air_velocity]]).reshape((1,-1))
    # vect= transformer.transform([[clo_insulation,metabolic_rate,air_temparature,relative_humidity,air_velocity]])
    prediction=classifier.predict(vect)
  
    # print(prediction)
    
    return prediction


    
def main():
    st.title("Thermal Comfort Prediction")
    html_temp = """
   <body style="background-color:#800080;">
      <div style="background-color:tomato;padding:10px">
      <h2 style="color:white;text-align:center;">Thermal Comfort Prediction ML App </h2>
    </div>
   </body>
    """
    st.markdown(html_temp,unsafe_allow_html=True) 
    
    # air_temparature=st.number_input ('Air Temperature (°C)',value=int(), min_value=0, max_value=50, step=1)
    # radiant_temparature=st.number_input ('Radiant Temperature (°C)',value=int(), min_value=0, step=1)
    # relative_humidity=st.number_input ('Relative Humidity (%)',value=int(), min_value=0, step=1)
    
    air_temparature=st.number_input ('Air Temperature (°C)',value=int())
    radiant_temparature=st.number_input ('Radiant Temperature (°C)',value=int())
    relative_humidity=st.number_input ('Relative Humidity (%)',value=int())
    air_velocity=st.number_input ('Air Velocity (m/s)', value=float())
    clo_insulation=st.number_input ('Clo Insulation (Clo)',value=float())
    metabolic_rate=st.number_input ('Metabolic Rate (Met)',value=float())
    result=""
    
    comfortable_html="""
    <div style="background-color:#F4D03F;padding:10px">
    <h2 style="color:white;text-align:center;">Thermally Comfortable </h2>
    </div>
    """
    slightly_comfortable_html="""
    <div style="background-color:#F00000;padding:10px">
    <h2 style="color:white;text-align:center;">Slightly Uncomfortable </h2>
    </div>
    """
    uncomfortable_html="""
    <div style="background-color:#F00000;padding:10px">
    <h2 style="color:white;text-align:center;">Thermally Uncomfortable </h2>
    </div>
    """
    
    
    if st.button("Predict"):
        result=thermal_comfort_prediction(clo_insulation,metabolic_rate,air_temparature,relative_humidity,air_velocity)
        # if (air_temparature < 8 ) or (air_temparature > 35.0):
        #     if result != 1 :
        #          result == 1
        st.success('Thermal Comfort value is {}'.format(result))     
        
        if result in [1,2]:
            st.markdown(uncomfortable_html,unsafe_allow_html=True)
        if result in [3,4] :
            st.markdown(slightly_comfortable_html,unsafe_allow_html=True)
        if result in [5,6] :
            st.markdown(comfortable_html,unsafe_allow_html=True)
          
          
    if st.button("About"):
        st.text("This is a thermal comfort prediction app which uses Machine Learning algorithm")
        st.text("Built on Python by using streamlit framework")
        st.text("By Mohammed Shareef")
          

if __name__=='__main__':
    main()
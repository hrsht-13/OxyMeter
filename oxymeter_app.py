import streamlit as st
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import IFrame
import pickle
import numpy as np

data=pd.read_csv("body_measurements.csv")
rad=st.sidebar.radio("Navigation",["Home","Dataset Info","Dashboard","Distributions","Important Ranges","Predict your Oxygen level","Error Analysis","About"])

if (rad=="Home"):
  header=st.beta_container()

  with header:
    st.title("OxyMeter Web App")
    st.header("Hello and Welcome to the oximeter web app.")
    photo=Image.open("images/home.jpg")
    st.image(photo,use_column_width=True)
    st.subheader("Know your Oxygen Saturation Level just by your Body Measurements.")


if (rad=="Dataset Info"):
  header=st.beta_container()

  with header:
    st.title("Information about Dataset")
    
    st.header("1. Dataset")
    st.write(data.head())

    st.header("2. Description")
    st.write(data.describe())

    st.header("3. Info")
    st.write(pd.DataFrame(data.dtypes,columns=["Dtype"]))

    st.header("4. Null Values")
    st.write(pd.DataFrame(data.isnull().sum(),columns=["null counts"]))

    st.header("5. Correlation")
    fig2 = plt.figure()
    sns.heatmap(data.corr("spearman"))
    st.pyplot(fig2)
    st.write(data.corr("spearman"))


if(rad=="Dashboard"):
  header=st.beta_container()

  with header:
    st.title("PowerBI Dashboard")
    powerBiEmbed = 'https://app.powerbi.com/reportEmbed?reportId=9c3576c1-44d1-4d8a-97fc-41368c5cc569&autoAuth=true&ctid=b9abe56c-43a7-4e67-a17b-32cfa05c95c8&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLWluZGlhLWNlbnRyYWwtYS1wcmltYXJ5LXJlZGlyZWN0LmFuYWx5c2lzLndpbmRvd3MubmV0LyJ9'
    st.write(IFrame(powerBiEmbed, width=800, height=600))
  

if(rad=="Distributions"):
  header=st.beta_container()
  
  left_col,right_col=st.beta_columns(2)

  with header:
    st.title("Distribution Plots")
    with left_col:
      fig = plt.figure(figsize=(6,5))
      sns.distplot(data["weight"])
      st.pyplot(fig)

      fig = plt.figure(figsize=(6,5))
      sns.distplot(data["diastolic_blood_pressure"])
      st.pyplot(fig)

      fig = plt.figure(figsize=(6,5))
      sns.distplot(data["heart_pulse"])
      st.pyplot(fig)

      fig = plt.figure(figsize=(6,5))
      sns.distplot(data["muscle_mass"])
      st.pyplot(fig)

      fig = plt.figure(figsize=(6,5))
      sns.distplot(data["hydration"])
      st.pyplot(fig)

      fig = plt.figure(figsize=(6,5))
      sns.distplot(data["w/h2"])
      st.pyplot(fig)


    with right_col:
      fig = plt.figure(figsize=(6,5))
      sns.distplot(data["height"])
      st.pyplot(fig)

      fig = plt.figure(figsize=(6,5))
      sns.distplot(data["systolic_blood_pressure"])
      st.pyplot(fig)

      fig = plt.figure(figsize=(6,5))
      sns.distplot(data["temperature"])
      st.pyplot(fig)

      fig = plt.figure(figsize=(6,5))
      sns.distplot(data["bone_mass"])
      st.pyplot(fig)

      fig = plt.figure(figsize=(6,5))
      sns.distplot(data["pulse_wave_velocity"])
      st.pyplot(fig)

      fig = plt.figure(figsize=(6,5))
      sns.distplot(data["muscle/bone"])
      st.pyplot(fig)


if (rad=="Important Ranges"):
  header=st.beta_container()

  with header:
    st.title("Check if your body measurements are in save range!")
    st.header("1. BMI Index")
    photo=Image.open("images/body-mass-index.jpg")
    st.image(photo,use_column_width=True)

    st.header("2. Temperature")
    photo=Image.open("images/temp.jpg")
    st.image(photo,use_column_width=True)

    st.header("3. Blood Pressure")
    photo=Image.open("images/press.jpg")
    st.image(photo,use_column_width=True)

    st.header("4. Oxygen Saturation")
    photo=Image.open("images/oxygen-saturation.jpg")
    st.image(photo,use_column_width=True)


if (rad=="Predict your Oxygen level"):
  header=st.beta_container()

  with header:
    st.title("Know your Oxygen Saturation Level")
    st.header("Enter your Body Measurements:")

    left_col,right_col=st.beta_columns(2)

    
    height=left_col.number_input("Height (cm)",)
    dia=left_col.number_input("Diastolic Blood Pressure (mmHg)",  )
    temp=left_col.number_input("Temperature (°C)",)
    muscle=(left_col.number_input("Muscle Mass (%)",))/100
    hyd=(left_col.number_input("Hydration (%)", ))/100
   
    weight=right_col.number_input("Weight (kg)", )
    sys=right_col.number_input("Systolic Blood Pressure (mmHg)",)
    pulse=right_col.number_input("Pulse Rate (BPM)", )
    bone=(right_col.number_input("Bone Mass (%)", ))/100
    vel=right_col.number_input("Pulse Wave Velocity (m/s)",)

    try:
      m_b=np.round(muscle/bone,2)
    except:
      m_b=0.0
    try:
      bmi=np.round(weight*0.1/(height*height*0.00001),2)
    except:
      bmi=0.0

    check_data=st.checkbox("View Your Values")
    if (check_data):
    # st.info("If you don't know your Muscle Mass, Bone Mass, Hydration level or Pulse Wave Velocity set it to ZERO")
      st.write("weight: ",weight)
      st.write("height: ",height)
      st.write("diastolic_blood_pressure: ",dia)
      st.write("systolic_blood_pressure: ",sys)
      st.write("heart_pulse: ",pulse)
      st.write("temperature: ",temp)
      st.write("muscle_mass: ",muscle)
      st.write("hydration: ",hyd)
      st.write("bone_mass: ",bone)
      st.write("pulse_wave_velocity: ",vel)
      st.write("muscle mass to bone mass: ",m_b)
      st.write("bmi value: ",bmi)

    predict=st.button("PREDICT SpO2")
    if(predict):
      lgb=pickle.load(open("final_model_lgb.sav", 'rb'))
      pred=lgb.predict([[weight,height,dia,sys,pulse,temp,muscle,hyd,bone, vel,bmi,m_b]])
      val=float(np.round(pred,2))
      result="Your Spo2 is: " + str(val) +"%"
      if(val<90):
        st.error(result)
      elif(val>=98):
        st.success(result)
      else:
        st.warning(result)


if (rad=="Error Analysis"):
    st.title("Model Evaluation and Error Analysis")

    st.header("1. Residuals")
    photo=Image.open("error_analysis/residual.jpg")
    st.image(photo,use_column_width=True)

    st.header("2. Prediction Error")
    photo=Image.open("error_analysis/prediction_error.jpg")
    st.image(photo,use_column_width=True)

    st.header("3. Cooks Distance")
    photo=Image.open("error_analysis/cook's distance.jpg")
    st.image(photo,use_column_width=True)

    st.header("4. Feature Selection")
    photo=Image.open("error_analysis/feature_selection.jpg")
    st.image(photo,use_column_width=True)

    st.header("5. Learning Curve")
    photo=Image.open("error_analysis/learning_curve.jpg")
    st.image(photo,use_column_width=True)

    st.header("6. Manifolds Learning")
    photo=Image.open("error_analysis/manifolds.jpg")
    st.image(photo,use_column_width=True)

    st.header("7. Validation Curve")
    photo=Image.open("error_analysis/validation.jpg")
    st.image(photo,use_column_width=True)

    st.header("8. Feature Importance")
    photo=Image.open("error_analysis/fea_imp1.jpg")
    st.image(photo,use_column_width=True)

    st.header("9. Interpret Model")
    photo=Image.open("error_analysis/interpret.jpg")
    st.image(photo,use_column_width=True)

    st.header("10. Model Prediction")
    st.subheader("With a train-test split of 70:30")
    photo=Image.open("error_analysis/prediction.jpg")
    st.image(photo,use_column_width=True)


if (rad=="About"):
  header=st.beta_container()

  with header:
    st.header("Information related to project:")
    st.markdown("* **Home**")
    st.write("Image: https://www.shutterstock.com/search/oxygen+in+blood")
    st.markdown("* **Dataset Info**")
    st.write("The dataset has been taken from https://www.kaggle.com/fsiamp/body-measurements")
    st.markdown("* **Dashboard** ")
    st.write("The dashboard has been created using **PowerBI**.")
    st.markdown("* **Distributions** ")
    st.write("Shows the population distribution of the feature.")
    st.markdown("* **Important Ranges** ")
    st.write("1. BMI Index: https://www.freepik.com/premium-vector/body-mass-index-vector-illustration_3790036.htm")
    st.write("2. Temperature: https://www.disabled-world.com/calculators-charts/degrees.php")
    st.write("3. Blood Pressure: https://www.healthline.com/health/high-blood-pressure-hypertension/blood-pressure-reading-explained")
    st.write("4. Oxygen Saturation: https://www.cosinuss.com/en/measured-data/vital-signs/oxygen-saturation/")
    st.markdown("* **Predict your Oxygen level**")
    st.write("Enter your body measurements to know your SpO2.")
    st.write("LGBMRegressor model has been prepared and analysed using **PyCaret**.")
    st.success("Built with :heart: by Harshita")




    





  
 

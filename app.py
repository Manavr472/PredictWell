import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

@st.cache_resource
def load_models():
    with open('models/RandomForest_Kidney_model.pkl', 'rb') as file:
        kidney_model = pickle.load(file)

    with open('models/Decision_tree_diabetes_model.pkl', 'rb') as file:
        diabetes_model = pickle.load(file)
        
    heart_model = keras.models.load_model('models/heart_disease_model.hdf5')

    return kidney_model, diabetes_model, heart_model

kidney_model, diabetes_model, heart_model = load_models()


# Streamlit app
def main():
    
    st.title('PredictWell: Chronic Disease Predictor')

    # Sidebar navigation
    st.sidebar.title('PredictWell')
    selected_disease = st.sidebar.radio('Select Disease:', ['Kidney Disease', 'Heart Disease', 'Diabetes'])
    
    healthy = Image.open("./cards/ok.png")
    unhealthy = Image.open("./cards/help.png")

    if selected_disease == 'Kidney Disease':
        st.subheader('Kidney Disease Prediction')
        # Add input fields and prediction logic for kidney disease prediction
        # Age input
        kd_age = st.slider("Age", min_value=0, max_value=100, value=30)

        gender = st.selectbox("Gender", ('Male','Female'))
        
        # Blood pressure input
        kd_bp = st.slider("Blood Pressure", min_value=50, max_value=180, value=70)

        # Albumin input
        kd_al = st.slider("Albumin", min_value=0, max_value=5, value=0)

        # Sugar input
        kd_su = st.slider("Sugar", min_value=0, max_value=5, value=0)

        # Bacteria input
        ba_options = ['Not Present', 'Present']
        k_ba = st.selectbox("Bacteria", ba_options)
        
        if k_ba == 'Not Present':
            kd_ba = 0
        elif k_ba == 'Present':
            kd_ba = 1
        

        # Blood glucose random input
        kd_bgr = st.slider("Blood Glucose", min_value=22.0, max_value=500.0, value=145.0)

        # Blood urea input
        kd_bu = st.slider("Blood Urea", min_value=0, max_value=400, value=56)

        # Serum creatinine input
        kd_sc = st.slider("Serum Creatinine", min_value=0.0, max_value=16.0, value=2.997)

        # Sodium input
        kd_sod = st.slider("Sodium", min_value=80, max_value=180, value=135)

        # Potassium input
        kd_pot = st.slider("Potassium", min_value=0, max_value=60, value=4)

        # Hemoglobin input
        kd_hemo = st.slider("Hemoglobin", min_value=0.0, max_value=25.0, value=12.5)

        # Packed cell volume input
        kd_pcv = st.slider("Packed Cell Volume", min_value=0.0, max_value=70.0, value=29.8)

        # White blood cell count input
        kd_wc = st.slider("White Blood Cell Count", min_value=3000, max_value=40000, value=6500)

        # Red blood cell count input
        kd_rc = st.slider("Red Blood Cell Count", min_value=0.0, max_value=10.0, value=4.5)

        # Hypertension input
        htn_options = ['No', 'Yes']
        k_htn = st.selectbox("Hypertension", htn_options)
        
        if k_htn == 'No':
            kd_htn = 0
        elif k_htn == 'Yes':
            kd_htn = 1

        # Diabetes mellitus input
        dm_options = ['No', 'Yes']
        k_dm = st.selectbox("Diabetes Mellitus", dm_options)
        
        if k_dm == 'No':
            kd_dm = 0
        elif k_dm == 'Yes':
            kd_dm = 1
            
        #Coronary Artery Disease input
        cad_options = ['No', 'Yes']
        k_cad = st.selectbox("Coronary Artery Disease", cad_options)
            
        if k_cad == 'No':
            kd_cad = 0
        elif k_cad == 'Yes':
            kd_cad = 1
            
        appet_options = ['Good', 'Poor']
        k_appet = st.selectbox('Appetite:', appet_options)
                
        if k_appet == 'Poor':
            kd_appet = 0
        elif k_appet == 'Good':
            kd_appet = 1
        
        def predict_kidney(kd_age,kd_bp,kd_al,kd_su,kd_ba,kd_bgr,kd_bu,kd_sc,kd_sod,kd_pot,kd_hemo,kd_pcv,kd_wc,kd_rc,kd_htn,kd_dm,kd_cad,kd_appet):
            input_data = np.array([[kd_age,kd_bp,kd_al,kd_su,kd_ba,kd_bgr,kd_bu,kd_sc,kd_sod,kd_pot,kd_hemo,kd_pcv,kd_wc,kd_rc,kd_htn,kd_dm,kd_cad,kd_appet]])
            data_2d = input_data.reshape(1, -1)
            kd_prediction = kidney_model.predict(data_2d)
            return kd_prediction
        
        if st.button("Predict"):
            kd_prediction = predict_kidney(kd_age,kd_bp,kd_al,kd_su,kd_ba,kd_bgr,kd_bu,kd_sc,kd_sod,kd_pot,kd_hemo,kd_pcv,kd_wc,kd_rc,kd_htn,kd_dm,kd_cad,kd_appet)
            print(kd_prediction)
            if kd_prediction == 0:
                st.write("Based on the input data, it is likely that you DO NOT HAVE Kidney Disease.")
                st.image(healthy, caption='healthy')
            elif kd_prediction == 1:
                st.write("Based on the input data, it is you HAVE Kidney Disease.")
                st.image(unhealthy, caption='unhealthy')
        
        

    elif selected_disease == 'Heart Disease':
        st.subheader('Heart Disease Prediction')
    #Heart diaseases prediction Starts
        # Add questions specific to Heart Disease
        age = st.slider("Age", min_value=1, max_value=100, value=30)
        gender = st.selectbox("Gender", ('Male','Female'))
    
        if gender == 'Male':
            sex = 0
        elif gender == 'Female':
            sex = 1
    
        cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])
        trestbps = st.slider("Resting Blood Pressure (trestbps)", min_value=90, max_value=200, value=120)
        chol = st.slider("Cholesterol (Chol)", min_value=100, max_value=600, value=200)
        fbs_value = st.slider("Fasting Blood Sugar (FBS)", min_value=0, max_value=400, value=120)

        if fbs_value < 120:
            fbs = 0
        else:
            fbs = 1

        restecg = st.selectbox("Resting Electrocardiographic Results (restECG)", [0, 1, 2])
        thalach = st.slider("Maximum Heart Rate Achieved (thalach)", min_value=70, max_value=420, value=150)
        exang = st.selectbox("Exercise-Induced Angina (exang)", [0, 1])
        oldpeak = st.slider("ST Depression (oldpeak)", min_value=0.0, max_value=6.2, value=1.0)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (Ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thallium Stress Test (thal)", [0, 1, 2, 3, 4, 5, 6, 7])
        # Add more questions as needed
    

        df = pd.read_csv("Heart_Disease_Prediction.csv")

        x = df.iloc[:, :-1]  # Select all columns except the last one (features)
        y = df.iloc[:, -1]  # Select the last column (target variable)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.35)

        def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            sc_x = StandardScaler()
            sc_x.fit(x_train)
            user_input_scaled = sc_x.transform(input_data)
            prediction = heart_model.predict(user_input_scaled)
            return prediction
        
        if st.button("Predict"):
            prediction = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
            print(prediction)
            if prediction >= 0.5:
                st.write("Based on the input data, it is likely that you HAVE heart disease.")
                st.image(unhealthy, caption='unhealthy')
            else:
                st.write("Based on the input data, it is likely that you DO NOT HAVE heart disease.")
                st.image(healthy, caption='healthy')

    #Heart diaseases prediction Ends

    elif selected_disease == 'Diabetes':
        st.subheader('Diabetes Prediction')
        # Add input fields and prediction logic for diabetes prediction
        db_gender = st.selectbox("Gender", ('Male','Female','Other'))
            
        if db_gender == 'Male':
            db_sex = 0
        elif db_gender == 'Female':
            db_sex = 1
        elif db_gender == 'Other':
            db_sex = 2
        
        db_age = st.slider("Age", min_value=1, max_value=100, value=30) 
        db_hyper = st.selectbox("Do you have Hypertension?", ('Yes','No'))
        
        if db_hyper == 'No':
            db_ht = 0
        elif db_hyper == 'Yes':
            db_ht = 1
        
        db_heart_diseases = st.selectbox("Do you have any Heart Diseases?", ('Yes','No'))
        
        if db_heart_diseases == 'No':
            db_hd = 0
        elif db_heart_diseases == 'Yes':
            db_hd = 1
            
        db_smoke = st.selectbox("Smoking Habits", ('Daily','Often','Occasionally','Quited','Never','Other'))
        
        if db_smoke == 'Daily':
            db_sh = 0
        elif db_smoke == 'Often':
            db_sh = 1
        elif db_smoke == 'Occasionally':
            db_sh = 2
        elif db_smoke == 'Quited':
            db_sh = 3
        elif db_smoke == 'Never':
            db_sh = 4
        elif db_smoke == 'Other':
            db_sh = 5
        
        db_bmi = st.slider("BMI(Body Mass Index)", min_value=0, max_value=100, value=24)
        db_hgb = st.slider("Hemoglobin A1C (HbA1c) levels(in %)", min_value=0.0, max_value=15.0, value=4.0)
        
        db_fbs = st.slider("Fasting Blood Sugar (FBS)", min_value=0, max_value=400, value=120)
        
        def predict_diabetes(db_sex, db_age, db_ht, db_hd, db_sh, db_bmi, db_hgb, db_fbs):
            input_data = np.array([[db_sex, db_age, db_ht, db_hd, db_sh, db_bmi, db_hgb, db_fbs]])
            data_2d = input_data.reshape(1, -1)
            db_prediction = diabetes_model.predict(data_2d)
            return db_prediction
        
        if st.button("Predict"):
            db_prediction = predict_diabetes(db_sex, db_age, db_ht, db_hd, db_sh, db_bmi, db_hgb, db_fbs)
            print(db_prediction)
            if db_prediction == 0:
                st.write("Based on the input data, it is likely that you DO NOT HAVE Diabetes.")
                st.image(healthy, caption='healthy')
            elif db_prediction == 1:
                st.write("Based on the input data, it is likely that you HAVE Diabetes.")
                st.image(unhealthy, caption='unhealthy')
                st.show()



if __name__ == '__main__':
    main()

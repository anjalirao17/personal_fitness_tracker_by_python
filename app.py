import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import base64
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")
    
def set_background():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #025d93;
        }
        [data-testid="stSidebar"] {
            background-color: rgba(0, 0, 0, 0.8);
        }
        .stButton > button {
            border-radius: 8px;
            background-color: #e3526e;
            color: white;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #ff0000;
        }

        .result-box {
            padding: 10px;
            border-radius: 10px;
            color: white;
            font-size: 18px;
            margin-bottom: 10px;
        }

        .calories-box{background-color: #86f4ee;
            color: black;
        }
        .bmi-box{background-color: #5d9302; }
        .diet-box{background-color: #FF9800;}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

#Loading and processing data
@st.cache_data 
def load_data():
    try:
        Fitness = pd.read_csv("fitness_dataset.csv").sample(frac=0.3, random_state=42)
    except FileNotFoundError:
        st.error("Enter: Data set file 'fitness_dataset.csv' not found." )
        return None, None
    
    Fitness.dropna(inplace=True) 
    
    #Remove outliers
    Q1 = Fitness["calories_burned"].quantile(0.25)
    Q3 = Fitness["calories_burned"].quantile(0.75)
    IQR=Q3-Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    Fitness = Fitness[(Fitness["calories_burned"] > lower_bound) & (Fitness["calories_burned"]< upper_bound)]

    #Feature Engineering
    Fitness["BMI"] = Fitness["weight_kg"] / ((Fitness["height_cm"]/100).replace(0,np.nan)**2)
    Fitness.dropna(inplace=True)

    #Encoder Categorical Variable
    encoder = LabelEncoder()
    if "activity_type" in Fitness.columns and "intensity" in Fitness.columns:
      Fitness["activity_type"] = encoder.fit_transform(Fitness["activity_type"])
      Fitness["intensity"] = encoder.fit_transform(Fitness["intensity"])

    return Fitness, encoder

Fitness, encoder = load_data()
if Fitness is None:
    st.stop()

# Train the model 
@st.cache_resource
def train_model():
    X = Fitness[['age', 'weight_kg', 'height_cm', 'BMI', 'activity_type', 'duration_minutes', 'intensity']]
    Y = Fitness['calories_burned']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)

    return model

model = train_model()

#BMI Category
def classify_bmi(BMI):
    if BMI < 18.5:
        return "Underweight"
    elif 18.5 <= BMI <= 24.9:
        return "Normal Weight"
    else:
        return "Overweight"
    
#Diet Plan Suggestion
def suggest_diet(BMI):
    if BMI < 18.5:
        return "Hight Calories Diet: Nuts and Seeds, Dairy Products, Rice, Avacado, Peanut Butter"
    elif 18.5 <= BMI <= 24.9:
        return "Balanced Deit: Fruits, Vegetables, Whole grains"
    else:
        return "Low Calories Deit: Green Veggies, Lean Protein, Avoid Sugar, Small Meals"
    
#App Creation    
st.title("Personal Fitness Tracker")
st.subheader("Calories Burned Prediction & Diet Plan Suggestion")
st.sidebar.header("Enter Your Details")

st.markdown("<br><br>",unsafe_allow_html=True)
placeholder = st.empty()
placeholder.info("Enter Your Details On The Sidebar & Click The Button To Get Strated!")

#user_input
age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=25)
weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, max_value=200.0, value=70.0)
height = st.sidebar.number_input("Height (cm)",min_value=50.0, max_value=250.0, value=170.0)
duration = st.sidebar.number_input("Duration of Activity (minutes)", min_value=1, max_value=300, value=30)
 
activity_type = st.sidebar.selectbox("Activity Type", options=["Running", "Walking", "Cycling", "Swimming"])
activity_map = {"Running":3, "Walking":1, "Cycling":2, "Swimming":4}
activity_val = activity_map.get(activity_type, 1)

intensity_level = st.sidebar.radio("Intensity Level",["Low", "Medium","High"])
intensity_map = {"Low":1, "Medium":2,"High":3}
intensity_val = intensity_map.get(intensity_level, 1)

if st.sidebar.button("Predict Calories & Suggest Deit Plan"):
    if height <= 0:
        st.error("Height must be greater than zero!")
    else:
        BMI = weight/((height/100)**2)
        bmi_category = classify_bmi(BMI)
        input_data = pd.DataFrame([[age, weight, height, BMI, activity_val, duration, intensity_val]],
                                  columns=['age', 'weight_kg', 'height_cm', 'BMI', 'activity_type', 'duration_minutes','intensity'])

        try:
          calories_pred = model.predict(input_data)[0]
          st.markdown(f"<br><br><div class='result-box calories-box'>Predicted Calories Burned: {calories_pred:.2f}</div>", unsafe_allow_html=True)

          st.subheader("**Your BMI Report**")
          st.markdown(f"<div class='result-box bmi-box'>Your BMI: {BMI:.2f} ({bmi_category})</div>", unsafe_allow_html=True)

          diet_suggestion = suggest_diet(BMI)
          st.subheader("**Recommended Diet Plan For You**")
          st.markdown(f"<div class='result-box diet-box'>Recommended Diet Plan: {diet_suggestion}</div>", unsafe_allow_html=True)
        except NotFittedError:
           st.error("Model training issue. Please check your dataset and preprocessing.")



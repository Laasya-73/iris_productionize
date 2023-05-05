import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pickle import load

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")

IMAGE_PATH = os.path.join(dir_of_interest, "images", "iris.jpg")
DATA_PATH = os.path.join(dir_of_interest, "data", "Iris.csv")

st.title("Welcome - IRIS Application")

img = image.imread(IMAGE_PATH)
st.image(img)

iris = pd.read_csv(DATA_PATH)
scaler = load(open('models/standard_scaler.pkl', 'rb'))
knn_classifier = load(open('models/knn_model.pkl', 'rb'))
lr_classifier = load(open('models/lr_model.pkl', 'rb'))
nb_classifier = load(open('models/nb_model.pkl', 'rb'))
dt_classifier = load(open('models/dt_model.pkl', 'rb'))
sv_classifier = load(open('models/sv_model.pkl', 'rb'))

st.header("Set the slider to see the Iris Flower Details")
sl = st.slider('Enter the SepalLength (in cm): ',0.0, 10.0, 5.0, 0.1,key="sl")
sw = st.slider('Enter the SepalWidth (in cm): ',0.0, 10.0, 4.0, 0.1,key="sw")
pl = st.slider('Enter the PetalLength (in cm): ',0.0, 10.0, 3.0, 0.1,key="pl")
pw = st.slider('Enter the PetalWidth (in cm): ',0.0, 10.0, 2.0, 0.1,key="pw")

query_point = np.array([sl, sw, pl, pw])
query_point.reshape(1, -1)
query_point = query_point.reshape(1, -1)
query_point_transformed = scaler.transform(query_point)

predict_k=knn_classifier.predict(query_point_transformed)
predict_l=lr_classifier.predict(query_point_transformed)
predict_n=nb_classifier.predict(query_point_transformed)
predict_d=dt_classifier.predict(query_point_transformed)
predict_s=sv_classifier.predict(query_point_transformed)

#st.header(f"The Iris Class is: {predict_k[0]}")
#st.header(f"The Iris Class is: {predict_l[0]}")
#st.header(f"The Iris Class is: {predict_d[0]}")
#st.header(f"The Iris Class is: {predict_s[0]}")
st.header(f"The Iris Class is: {predict_n[0]}")



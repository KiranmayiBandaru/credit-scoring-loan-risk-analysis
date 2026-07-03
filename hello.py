import streamlit as st
import pandas as pd
import joblib

st.write("app started successfuly")
st.write("second line of page")
model = joblib.load("xgb_model.pkl")
print(type(model))
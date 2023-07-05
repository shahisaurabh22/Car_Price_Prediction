import streamlit as st
import numpy as np
import pickle
from sklearn import *

pipe = pickle.load(open('pipe_RF.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title('Used Car Price Calculator 🚔')

company = st.selectbox('Brand',df['company'].unique())
name = st.selectbox('Model',df['name'].unique())
fuel = st.selectbox('Fuel Type',df['fuel'].unique())
seller_type = fuel = st.selectbox('Seller',df['seller_type'].unique())
transmission = st.selectbox('Transmission',df['transmission'].unique())
km_driven = st.number_input('KM Driven',100,600000,step=50)
owner = st.selectbox('Owner',df['owner'].unique())
year_old_2023 = st.number_input('Year Old',0,50)

if st.button('Look What You Selected'):
    st.write('You have selected', {'Brand' : company, 'Model' : name, 'Fuel Type' : fuel, 'Seller': seller_type, 'Transmission' : transmission,
                               'KM Driven' : km_driven, 'Owner' : owner, 'Year Old' : year_old_2023 })


if st.button('Calculate the Price'):
    query = np.array([name,km_driven,fuel,seller_type,transmission,owner,company,year_old_2023])
    query = query.reshape(1,8)

    st.title("The predicted price is " + str(np.round(pipe.predict(query)[0], 2)))
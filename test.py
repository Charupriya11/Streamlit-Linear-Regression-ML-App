import streamlit as st
from sklearn import linear_model
import numpy as np
import pandas as pd
import warnings
import pickle

warnings.filterwarnings("ignore")

data = pd.read_csv("cattle.csv")
X = data[['weight','age']]
y = data['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)
pickle.dump(regr,open('model.pkl','wb'))

pickle.dump(regr,open('model.pkl','wb'))

model=pickle.load(open("model.pkl","rb"))


def predict_forest(weight,age):
    prediction=regr.predict([[weight, age]])
    return prediction

def main():
    st.title("LIVESTOCK MANAGEMENT")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Milk Yield Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    breed = st.selectbox('BREED TYPE :', ('Gir', 'Sahiwal', 'Red Sindhi', 'Rathi', 'Kangayam'))
    weight = st.number_input('Weight :')
    age = st.number_input('Age :')
    capacity = st.selectbox('CAPACITY :', ('Excellent', 'Good', 'Average'))

    if st.button("Predict"):
        output=predict_forest(weight,age)
        st.success('MILK YIELD PREDICTION : {} litres per day'.format(output))

if __name__=='__main__':
    main()





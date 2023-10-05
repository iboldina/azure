import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import streamlit.components.v1 as components

st.title('Modèle de scoring')
st.subheader("Prédiction sur la probabilité de faillite d'un client")
st.write(("Dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle"))


#import our data
data = pd.read_csv('test_production.csv')
st.write(data.head())

# Load the model
model = joblib.load('model.joblib')# on recupere pas le modele

# Ask the user to input an ID
id = st.number_input('Please enter an ID', min_value=1, step=1)

def predict(id: int):
    # Select the row where SK_ID_CURR equals the provided id
    row = data[data['SK_ID_CURR'] == id]
    # Use the model to make a prediction on this row
    prediction = model.predict(row) #remplacer par un appel via api request.pst, url de mon api(localhost pour instant)
    log_proba = model.predict_proba(row)

    # Get the maximum probability
    max_log_proba = np.max(log_proba)


 # Transform the result into words
    result = 'Prêt OK' if prediction[0] == 0 else 'Prêt Not OK'
    return result, max_log_proba# Call the predict function when the user inputs an ID

if st.button('Predict'):
    result, max_log_proba = predict(id)
    st.write(result)
    st.write("The maximum predicted probability is: {:.2f}".format(max_log_proba))

# Ask the user to input a column name
column = st.selectbox('Please select a column', data.columns[1:])


def plot_distribution(column: str, id: int):
    # Plot the distribution of the selected column
    plt.hist(data[column], bins=30, edgecolor='black')
    plt.title('Distribution of ' + column)
    plt.xlabel(column)
    plt.ylabel('Frequency')

    # Add a vertical line at the value of the selected ID
    plt.axvline(x=data.loc[data['SK_ID_CURR'] == id, column].values[0], color='r', linestyle='--')

# Call the plot_distribution function when the user selects a column
if st.button('Show Distribution'):
    plot_distribution(column, id)
    st.pyplot(plt)




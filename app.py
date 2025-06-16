import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('bank_marketing_model.pkl')

# Define feature lists (same as in the notebook)
numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                   'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                   'euribor3m', 'nr.employed']
cat_columns = ['job', 'marital', 'education', 'default', 'housing', 
               'loan', 'contact', 'month', 'day_of_week', 'poutcome']

# Define categorical options (based on df[col].unique() from notebook)
categorical_options = {
    'job': ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
            'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'],
    'marital': ['divorced', 'married', 'single'],
    'education': ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
                  'illiterate', 'professional.course', 'university.degree'],
    'default': ['no', 'yes'],
    'housing': ['no', 'yes'],
    'loan': ['no', 'yes'],
    'contact': ['cellular', 'telephone'],
    'month': ['apr', 'aug', 'dec', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep'],
    'day_of_week': ['fri', 'mon', 'thu', 'tue', 'wed'],
    'poutcome': ['failure', 'nonexistent', 'success']
}

# Streamlit app layout
st.title("Bank Marketing Term Deposit Prediction")
st.write("Enter client and campaign details to predict if they will subscribe to a term deposit.")

# Create input fields
st.subheader("Numeric Features")
numeric_inputs = {}
for col in numeric_columns:
    if col == 'age':
        numeric_inputs[col] = st.slider(col, min_value=18, max_value=100, value=40)
    elif col == 'duration':
        numeric_inputs[col] = st.number_input(col, min_value=0, value=200, help="Call duration in seconds")
    elif col == 'campaign':
        numeric_inputs[col] = st.number_input(col, min_value=1, value=1, help="Number of contacts this campaign")
    elif col == 'pdays':
        numeric_inputs[col] = st.number_input(col, min_value=0, value=999, help="Days since last contact (999 if not contacted)")
    elif col == 'previous':
        numeric_inputs[col] = st.number_input(col, min_value=0, value=0, help="Number of previous contacts")
    else:
        numeric_inputs[col] = st.number_input(col, value=0.0, format="%.3f")

st.subheader("Categorical Features")
categorical_inputs = {}
for col in cat_columns:
    categorical_inputs[col] = st.selectbox(col, options=categorical_options[col])

# Predict button
if st.button("Predict"):
    # Combine inputs into a DataFrame
    input_data = {**numeric_inputs, **categorical_inputs}
    input_df = pd.DataFrame([input_data])

    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of "yes"

        # Display results
        st.subheader("Prediction Result")
        if prediction == 'yes':
            st.success(f"The client is likely to subscribe to a term deposit (Probability: {probability:.2%})")
        else:
            st.warning(f"The client is unlikely to subscribe to a term deposit (Probability: {probability:.2%})")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Footer
st.markdown("---")
st.write("Built with Streamlit. Model trained on bank-additional-full dataset.")
import streamlit as st
import pickle
import pandas as pd
# Add necessary scikit-learn imports for the pickled model
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Define the filename for the pickle file
filename = 'decision_tree_model.pkl' # Updated filename to match the saved pruned model
data_info_filename = 'data_info.pkl'

# Load the model from the pickle file
with open(filename, 'rb') as file:
    model = pickle.load(file)

# Load data info (including expected columns and categorical values)
with open(data_info_filename, 'rb') as file:
    data_info = pickle.load(file)

expected_columns = data_info['expected_columns']
categorical_unique_values = data_info['categorical_unique_values']

st.title('Credit Default Prediction')

st.header('Enter Customer Details:')

# Define input widgets for numerical features based on observed ranges in the data
# Assuming 'default' dataframe is available or its characteristics are known.
# Based on the previous notebook cells, the original numerical columns were:
# months_loan_duration, amount, percent_of_income, years_at_residence, age
# Using hardcoded reasonable ranges as the original 'default' df is not available in app.py context
months_loan_duration = st.slider('Months Loan Duration', min_value=6, max_value=72, value=12)
amount = st.slider('Amount', min_value=250, max_value=18424, value=1000)
percent_of_income = st.slider('Percent of Income', min_value=1, max_value=4, value=2)
years_at_residence = st.slider('Years at Residence', min_value=1, max_value=4, value=2)
age = st.slider('Age', min_value=18, max_value=75, value=30)

# Assuming 'existing_loans_count' and 'dependents' are also needed as per the original data
# You might want to add input widgets for these as well if they are not constant
existing_loans_count = st.slider('Number of Existing Loans', min_value=1, max_value=4, value=1)
dependents = st.slider('Number of Dependents', min_value=1, max_value=2, value=1)


st.subheader('Categorical Features')

checking_balance = st.selectbox('Checking Balance', categorical_unique_values['checking_balance'])
credit_history = st.selectbox('Credit History', categorical_unique_values['credit_history'])
purpose = st.selectbox('Purpose', categorical_unique_values['purpose'])
savings_balance = st.selectbox('Savings Balance', categorical_unique_values['savings_balance'])

# Handle employment_duration mapping for the Streamlit app
employment_duration_mapping = {
    'unemployed': 0,
    '< 1 yr': 1,
    '1–4 yrs': 2,
    '4–7 yrs': 3,
    '≥ 7 yrs': 4
}
# Reverse mapping for selectbox options (using descriptive labels)
employment_duration_reverse_mapping = {v: k for k, v in employment_duration_mapping.items()}
employment_duration_options = list(employment_duration_reverse_mapping.keys())
employment_duration_label = st.selectbox('Employment Duration', employment_duration_options)
# Map the selected label back to the numerical value
employment_duration = employment_duration_mapping[employment_duration_label]


other_credit = st.selectbox('Other Credit', categorical_unique_values['other_credit'])
housing = st.selectbox('Housing', categorical_unique_values['housing'])
job = st.selectbox('Job', categorical_unique_values['job'])
phone = st.selectbox('Phone', categorical_unique_values['phone'])

# Collect user input into a dictionary
user_input = {
    'months_loan_duration': months_loan_duration,
    'amount': amount,
    'percent_of_income': percent_of_income,
    'years_at_residence': years_at_residence,
    'age': age,
    'existing_loans_count': existing_loans_count,
    'dependents': dependents,
    'checking_balance': checking_balance,
    'credit_history': credit_history,
    'purpose': purpose,
    'savings_balance': savings_balance,
    'employment_duration': employment_duration, # Use the numerical mapping
    'other_credit': other_credit,
    'housing': housing,
    'job': job,
    'phone': phone
}

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Apply one-hot encoding to categorical features (excluding employment_duration as it's already encoded)
categorical_cols_for_ohe = ['checking_balance', 'credit_history', 'purpose', 'savings_balance', 'other_credit', 'housing', 'job', 'phone']
input_encoded = pd.get_dummies(input_df, columns=categorical_cols_for_ohe, drop_first=True, dtype=int)

# Ensure all columns from training data are present and in the same order
for col in expected_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_encoded = input_encoded[expected_columns]


# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_encoded)
    prediction_proba = model.predict_proba(input_encoded)

    st.subheader('Prediction Result:')
    if prediction[0] == 0:
        st.success('Prediction: No Default')
        st.write(f'Probability of No Default: {prediction_proba[0][0]:.2f}')
        st.write(f'Probability of Default: {prediction_proba[0][1]:.2f}')
    else:
        st.error('Prediction: Default')
        st.error(f'Probability of Default: {prediction_proba[0][1]:.2f}') # Changed to st.error for visibility
        st.write(f'Probability of No Default: {prediction_proba[0][0]:.2f}')

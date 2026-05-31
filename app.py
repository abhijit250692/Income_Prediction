import streamlit as st
import joblib
import numpy as np

# Load the best model
model = joblib.load('XGBClassifier_model.pkl')
Workclass_map = {'Federal-gov': 0, 'Local-gov': 1, 'Never-worked': 2, 'Private': 3, 'Self-emp-inc': 4, 'Self-emp-not-inc': 5, 'State-gov': 6, 'Without-pay': 7}
Education_map = {'10th': 0, '11th': 1, '12th': 2, '1st-4th': 3, '5th-6th': 4, '7th-8th': 5, '9th': 6, 'Assoc-acdm': 7, 'Assoc-voc': 8, 'Bachelors': 9, 'Doctorate': 10, 'HS-grad': 11, 'Masters': 12, 'Preschool': 13, 'Prof-school': 14, 'Some-college': 15}
Marital_Status_map = {'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2, 'Married-spouse-absent': 3, 'Never-married': 4, 'Separated': 5, 'Widowed': 6}
Occupation_map = {'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2, 'Exec-managerial': 3, 'Farming-fishing': 4, 'Handlers-cleaners': 5, 'Machine-op-inspct': 6, 'Other-service': 7, 'Priv-house-serv': 8, 'Prof-specialty': 9, 'Protective-serv': 10, 'Sales': 11, 'Tech-support': 12, 'Transport-moving': 13}
Sex_map = {'Female': 0, 'Male': 1}
Income_map = {'<=50K': 0, '>50K': 1}
image_url = "https://miro.medium.com/v2/resize:fit:1200/0*hpFoHzWYliOkGrOO.jpg"


def main():
    st.title("Income Prediction", text_alignment="center")
    st.markdown(
        f'<img src="{image_url}" style="width:800px; height:350px; object-fit:cover;">',
        unsafe_allow_html=True
    )

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider('Age', min_value=15, max_value=100, value=30, step=1)
            workclass = st.selectbox('Workclass', options=list(Workclass_map.keys()))
            education = st.selectbox('Education', options=list(Education_map.keys()))
            maritalStatus = st.selectbox('Marital Status', options=list(Marital_Status_map.keys()))
            occupation = st.selectbox('Occupation', options=list(Occupation_map.keys()))
        with col2:
            sex = st.radio("Sex", options=list(Sex_map.keys()))
            capitalGain = st.number_input('Capital Gain (USD)', min_value=0, max_value=100000, value=1000, step=1)
            capitalLoss = st.number_input('Capital Loss (USD)', min_value=0, max_value=4500, value=100, step=1)
            hoursPerWeek = st.number_input('Work Hours / Week', min_value=0, max_value=100, value=40, step=1)
        lcol1, lcol2, lcol3 = st.columns([2, 1, 2])
        with lcol2:
            submitted = st.form_submit_button("Predict Income")
    if submitted:
        input_data = np.array([[age, Workclass_map[workclass], Education_map[education], Marital_Status_map[maritalStatus], Occupation_map[occupation], 
                                Sex_map[sex], capitalGain, capitalLoss, hoursPerWeek]])
        predictedIncome = model.predict(input_data)
        if predictedIncome[0] == 1:
            st.success("Predicted income > 50K USD")
        else:
            st.success("Predicted income <= 50K USD")

if __name__ == "__main__":
    main()

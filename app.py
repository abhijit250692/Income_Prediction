import streamlit as st
import joblib
import numpy as np

# Load the best model
model = joblib.load('XGBClassifier_model.pkl')
scaler = joblib.load('scaler.pkl')

Workclass_map ={'Federal-Gov': 0, 'Local-Gov': 1, 'Private': 2, 'Self-Emp-Inc': 3, 'Self-Emp-Not-Inc': 4, 'State-Gov': 5, 'Without-Pay': 6}
Education_map = {'10Th': 0, '11Th': 1, '12Th': 2, '1St-4Th': 3, '5Th-6Th': 4, '7Th-8Th': 5, '9Th': 6, 'Assoc-Acdm': 7, 'Assoc-Voc': 8, 'Bachelors': 9, 'Doctorate': 10, 'Hs-Grad': 11, 'Masters': 12, 'Preschool': 13, 'Prof-School': 14, 'Some-College': 15}
Marital_Status_map = {'Divorced': 0, 'Married-Af-Spouse': 1, 'Married-Civ-Spouse': 2, 'Married-Spouse-Absent': 3, 'Never-Married': 4, 'Separated': 5, 'Widowed': 6}
Occupation_map = {'Adm-Clerical': 0, 'Armed-Forces': 1, 'Craft-Repair': 2, 'Exec-Managerial': 3, 'Farming-Fishing': 4, 'Handlers-Cleaners': 5, 'Machine-Op-Inspct': 6, 'Other-Service': 7, 'Priv-House-Serv': 8, 'Prof-Specialty': 9, 'Protective-Serv': 10, 'Sales': 11, 'Tech-Support': 12, 'Transport-Moving': 13}
Race_map = {'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1, 'Black': 2, 'Other': 3, 'White': 4}
Sex_map = {'Female': 0, 'Male': 1}
Native_Country_map = {'Cambodia': 0, 'Canada': 1, 'China': 2, 'Columbia': 3, 'Cuba': 4, 'Dominican-Republic': 5, 'Ecuador': 6, 'El-Salvador': 7, 'England': 8, 'France': 9, 'Germany': 10, 'Greece': 11, 'Guatemala': 12, 'Haiti': 13, 'Holand-Netherlands': 14, 'Honduras': 15, 'Hong': 16, 'Hungary': 17, 'India': 18, 'Iran': 19, 'Ireland': 20, 'Italy': 21, 'Jamaica': 22, 'Japan': 23, 'Laos': 24, 'Mexico': 25, 'Nicaragua': 26, 'Outlying-Us(Guam-Usvi-Etc)': 27, 'Peru': 28, 'Philippines': 29, 'Poland': 30, 'Portugal': 31, 'Puerto-Rico': 32, 'Scotland': 33, 'South': 34, 'Taiwan': 35, 'Thailand': 36, 'Trinadad&Tobago': 37, 'United-States': 38, 'Vietnam': 39, 'Yugoslavia': 40}

image_url = "https://miro.medium.com/v2/resize:fit:1200/0*hpFoHzWYliOkGrOO.jpg"

def prepare_input(p_age, p_workclass, p_education, p_maritalStatus, p_occupation, p_race, p_sex, p_capitalGain, p_capitalLoss, p_hoursPerWeek, p_nativeCountry):
    
    l_capitalGain = np.log1p(p_capitalGain)
    l_capitalLoss = np.log1p(p_capitalLoss)

    input_data = np.array([[np.float64(p_age), p_workclass, p_education, p_maritalStatus, p_occupation, p_race, 
                            p_sex, np.float64(l_capitalGain), np.float64(l_capitalLoss), p_hoursPerWeek, p_nativeCountry]])
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

def main():
    st.title("Income Prediction", text_alignment="center")
    st.markdown(
        f'<img src="{image_url}" style="width:800px; height:350px; object-fit:cover;">',
        unsafe_allow_html=True
    )

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider('Age', min_value=18, max_value=80, value=30, step=1)
            workclass = st.selectbox('Workclass', options=list(Workclass_map.keys()))
            education = st.selectbox('Education', options=list(Education_map.keys()))
            maritalStatus = st.selectbox('Marital Status', options=list(Marital_Status_map.keys()))
            occupation = st.selectbox('Occupation', options=list(Occupation_map.keys()))
            race = st.selectbox('Race', options=list(Race_map.keys()))
        with col2:
            sex = st.radio("Sex", options=list(Sex_map.keys()))
            capitalGain = st.number_input('Capital Gain (USD)', min_value=0, max_value=100000, value=1000, step=1)
            capitalLoss = st.number_input('Capital Loss (USD)', min_value=0, max_value=4500, value=100, step=1)
            hoursPerWeek = st.number_input('Work Hours / Week', min_value=1, max_value=100, value=40, step=1)
            nativeCountry = st.selectbox('Native Country', options=list(Native_Country_map.keys()))
        lcol1, lcol2, lcol3 = st.columns([2, 1, 2])
        with lcol2:
            submitted = st.form_submit_button("Predict")
    if submitted:
        input_data_scaled = prepare_input(age, Workclass_map[workclass], Education_map[education], Marital_Status_map[maritalStatus], Occupation_map[occupation], Race_map[race],
                                Sex_map[sex], capitalGain, capitalLoss, hoursPerWeek, Native_Country_map[nativeCountry])
        predicted_income = model.predict(input_data_scaled)[0]
        if predicted_income == 1:
            st.success("Predicted income > 50K USD")
        else:
            st.success("Predicted income <= 50K USD")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# View
st.set_page_config(page_title="Lung Cancer Prediction", layout="wide")
st.title('Lung Cancer Prediction')
st.write('''
The Lung Cancer Prediction machine learning model is developed to predict the likelihood of an individual having lung cancer based on various demographic and health - related factors. The dataset used for training and testing the model is obtained from Kaggle, specifically from the following source: https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer
         ''')
st.image(image=Image.open('Lung Cancer.jpg'), width=350)

st.header('Upload CSV File')
uploadedFile = st.file_uploader('', type=['csv'])
if uploadedFile is not None:
    inputData = pd.read_csv(uploadedFile)
else:
    st.header('User Input')

    def inputProcessing():
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox('Gender', ('Male', 'Female'))
            smoke = st.radio('Do you smoke?', ('Yes', 'No'))
            yellowFinger = st.radio(
                'Have you noticed any yellow discoloration specifically on your fingers or nails?', ('Yes', 'No'))
            anxiety = st.radio(
                'Have you experienced anxiety specifically related to your breathing or health?', ('Yes', 'No'))
            peerPressure = st.radio(
                'Have you experienced any pressure from peers or family members to engage in smoking or other behaviors that may affect your lung health?', ('Yes', 'No'))
            chronicDisease = st.radio(
                'Do you have a history of chronic respiratory conditions such as asthma or COPD?', ('Yes', 'No'))
            fatigue = st.radio(
                'Have you experienced unexplained fatigue or weakness, especially accompanied by other respiratory symptoms?', ('Yes', 'No'))
            allergy = st.radio(
                'Do you have any allergies, particularly to substances that may affect lung health?', ('Yes', 'No'))
        with col2:
            age = st.number_input('Age', min_value=10, max_value=100)
            wheezing = st.radio(
                'Have you experienced wheezing, particularly when breathing in or out?', ('Yes', 'No'))
            alcohol = st.radio(
                'Do you consume alcohol regularly?', ('Yes', 'No'))
            cough = st.radio(
                'Have you experienced persistent coughing, especially if it produces blood or lasts for several weeks?', ('Yes', 'No'))
            shortBreath = st.radio(
                'Have you experienced shortness of breath that is worsening over time or is not relieved by rest?', ('Yes', 'No'))
            swallow = st.radio(
                'Do you have any difficulty swallowing, particularly if accompanied by other respiratory symptoms?', ('Yes', 'No'))
            chestPain = st.radio(
                'Have you experienced chest pain or tightness, especially if it is persistent or worsens with breathing or exertion?', ('Yes', 'No'))

        mappingCategory = {'Yes': 1, 'No': 0}

        data = {
            'GENDER': 1 if gender == 'Male' else 0,
            'AGE': age,
            'SMOKING': mappingCategory.get(smoke),
            'YELLOW_FINGERS': mappingCategory.get(yellowFinger),
            'ANXIETY': mappingCategory.get(anxiety),
            'PEER_PRESSURE': mappingCategory.get(peerPressure),
            'CHRONIC DISEASE': mappingCategory.get(chronicDisease),
            'FATIGUE': mappingCategory.get(fatigue),
            'ALLERGY': mappingCategory.get(allergy),
            'WHEEZING': mappingCategory.get(wheezing),
            'ALCOHOL CONSUMING': mappingCategory.get(alcohol),
            'COUGHING': mappingCategory.get(cough),
            'SHORTNESS OF BREATH': mappingCategory.get(shortBreath),
            'SWALLOWING DIFFICULTY': mappingCategory.get(swallow),
            'CHEST PAIN': mappingCategory.get(chestPain),
        }
        features = pd.DataFrame(data, index=[0])
        return features


inputData = inputProcessing()

if st.button('Predict'):
    data = inputData

    with open('lung_cancer.pkl', 'rb') as file:
        pick = pickle.load(file)

    predictions = pick.predict(data)
    for prediction in predictions:
        st.subheader('Prediction Result :')
        with st.spinner('Analyzing the provided input...'):
            time.sleep(2)
            if prediction == 1:
                st.warning(
                    '🚨 Attention : The analysis suggests a potential risk of Lung Cancer. Please consult with a healthcare professional for further evaluation!')
            else:
                st.success(
                    '✅ Good news! The analysis indicates no significant signs of Lung Cancer based on the provided input')

# Data Processing
data = pd.read_csv('Data Processing.csv')

X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
smote = SMOTE()
X_oversampled, y_oversampled = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier()
paramGrid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False],
    'random_state': [42]
}
gridSearch = GridSearchCV(model, paramGrid, cv=5)
gridSearch.fit(X_oversampled, y_oversampled)
bestModel = gridSearch.best_estimator_
bestModel.fit(X_oversampled, y_oversampled)

pklname = 'lung_cancer.pkl'
with open(pklname, 'wb') as file:
    pickle.dump(bestModel, file)

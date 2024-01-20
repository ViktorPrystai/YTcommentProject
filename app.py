import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your data
data = pd.read_csv('youtoxic_english.csv')

# Processed data
new_data = data["Text"]
target = data["IsToxic"]

# Load the vectorizer
vectorizer = joblib.load("myvectorizer.joblib")

# Load the trained balanced model
loaded_classifier = joblib.load('RandomForestClassifier_balanced_model.joblib')

# Streamlit app
def classify_string(string):
    clf_name = "RandomForestClassifier"
    prediction = loaded_classifier.predict(vectorizer.transform([string]).toarray())[0]

    if prediction == 0:
        result = f"{clf_name} : НЕ ТОКСИЧНИЙ : {string}"
    else:
        result = f"{clf_name} :  ТОКСИЧНИЙ : {string}"

    return result

def main():
    # Add chat icon or message to the title
    st.title("Comment Toxicity Classification App 🗨️")

    # User input for comment
    user_input = st.text_input("Введіть ваш коментар:")

    if st.button("Перевірити"):
        if user_input:
            # Classification
            result = classify_string(user_input)
            st.write(result)
        else:
            st.warning("Будь ласка, введіть коментар.")

if __name__ == "__main__":
    main()






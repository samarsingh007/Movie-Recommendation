import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def naive_bayes():
    st.title("Naive Bayes Model for Movie Gross Prediction")

    # File Uploader
    uploaded_file = st.file_uploader("Upload your dataset (Excel file)", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
    else:
        # Default dataset
        df = pd.read_excel("src/Dataset/Movie_Gross_Value.xlsx")

    # Data preprocessing
    genre_split = df['GENRE'].str.split(', ', expand=True)
    genre_split.columns = [f'genre{i + 1}' for i in range(genre_split.shape[1])]
    df_split = pd.concat([df, genre_split], axis=1)
    
    # Feature Engineering and Encoding
    features = df_split[['genre1', 'genre2', 'genre3', 'VOTES', 'RATING']]
    label = df_split['Gross']
    encoder = OneHotEncoder(sparse=False, drop='first')
    genre_encoded = encoder.fit_transform(features[['genre1', 'genre2', 'genre3']])
    genre_encoded_df = pd.DataFrame(genre_encoded, columns=encoder.get_feature_names_out(['genre1', 'genre2', 'genre3']))
    features_encoded = pd.concat([genre_encoded_df, df_split[['VOTES', 'RATING']]], axis=1)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, label, test_size=0.2, random_state=42)

    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Model Evaluation
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix Visualization
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

def main():
    naive_bayes()

if __name__ == "__main__":
    main()

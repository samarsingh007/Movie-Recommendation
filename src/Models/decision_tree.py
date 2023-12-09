import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Function to categorize ratings
def categorize_rating(rating):
    if rating >= 7.0:
        return 'High'
    elif rating < 4.0:
        return 'Low'
    else:
        return 'Medium'

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Streamlit main function
def decision_tree_main():
    st.title("Decision Tree Classifier for Movie Ratings")

    # Default dataset path
    default_dataset_path = 'src/Dataset/movies.csv'
    movies_df = pd.read_csv(default_dataset_path)

    # Option to upload different dataset
    st.subheader("Upload Your Own Dataset (Optional)")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        movies_df = pd.read_csv(uploaded_file)

    # Proceed with the loaded dataset
    # Data Preprocessing
    movies_df['YEAR'] = movies_df['YEAR'].str.extract('(\d{4})').astype(float)
    movies_df['VOTES'] = movies_df['VOTES'].str.replace(',', '').astype(float)
    features = movies_df[['YEAR', 'RunTime', 'VOTES']]
    movies_df['RatingCategory'] = movies_df['RATING'].apply(categorize_rating)
    target_categorical = movies_df['RatingCategory'].fillna('Medium')

    imputer = SimpleImputer(strategy='median')
    features_imputed = imputer.fit_transform(features)

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_imputed)

    # Model Training
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        features_scaled, target_categorical, test_size=0.2, random_state=42)

    decision_tree_classifier = DecisionTreeClassifier(random_state=42)
    decision_tree_classifier.fit(X_train_cat, y_train_cat)

    # Predicting and Evaluating
    y_pred_cat = decision_tree_classifier.predict(X_test_cat)
    accuracy = accuracy_score(y_test_cat, y_pred_cat)
    classification_rep = classification_report(y_test_cat, y_pred_cat)

    # Displaying Results
    st.write(f"Accuracy: {accuracy}")
    st.text("Classification Report")
    st.text(classification_rep)

    # Confusion Matrix
    cnf_matrix = confusion_matrix(y_test_cat, y_pred_cat)
    plot_confusion_matrix(cnf_matrix, classes=['High', 'Medium', 'Low'])
    st.pyplot(plt)

if __name__ == "__main__":
    decision_tree_main()

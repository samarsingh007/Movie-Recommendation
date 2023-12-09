import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import itertools

def logistic_regression():
    st.title("Logistic Regression for Movie Ratings")

    # File Uploader
    uploaded_file = st.file_uploader("Upload your dataset (Optional)", type=["csv"])
    if uploaded_file is not None:
        movies_df = pd.read_csv(uploaded_file)
    else:
        # Default dataset
        csv_file_path = 'src/Dataset/movies.csv'
        movies_df = pd.read_csv(csv_file_path)

    # Clean data
    movies_df = movies_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Convert 'VOTES' to numerical format and handle missing values
    movies_df['VOTES'] = movies_df['VOTES'].str.replace(',', '').astype(float)
    movies_df['VOTES'] = movies_df['VOTES'].fillna(movies_df['VOTES'].median())

    # Create a binary target variable for 'highly rated' (rating >= 7.0)
    movies_df['HIGHLY_RATED'] = (movies_df['RATING'] >= 7.0).astype(int)

    # Feature Selection
    features = ['YEAR', 'GENRE', 'VOTES', 'RunTime']
    target = 'HIGHLY_RATED'

    # Preprocessing for categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), ['VOTES', 'RunTime']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['GENRE'])
        ])

    # Pipeline with preprocessing and logistic regression model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', LogisticRegression())
    ])

    # Split data into training and testing sets
    X = movies_df[features]
    y = movies_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the logistic regression model
    model_pipeline.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model_pipeline.predict(X_test)
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Visualization: Coefficients Plot
    st.subheader("Top Logistic Regression Coefficients")

    coefficients = model_pipeline.named_steps['classifier'].coef_[0]
    feature_names = model_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out()
    numeric_features = ['VOTES', 'RunTime']
    all_features = numeric_features + list(feature_names)

    # Creating a dictionary of feature names and their coefficients
    coefficients_dict = dict(zip(all_features, coefficients))

    top_n = 20
    sorted_coefficients = dict(sorted(coefficients_dict.items(), key=lambda item: abs(item[1]), reverse=True))
    top_features = list(sorted_coefficients.keys())[:top_n]
    top_coefficients = [sorted_coefficients[feature] for feature in top_features]

    # Plotting top N coefficients
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(range(len(top_features)), top_coefficients, align='center')
    ax.set_xticks(range(len(top_features)))
    ax.set_xticklabels(top_features, rotation='vertical')
    ax.set_xlabel('Features')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(f'Top {top_n} Logistic Regression Coefficients')
    st.pyplot(fig)

    # Visualization: Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(['Not Highly Rated', 'Highly Rated'], rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(['Not Highly Rated', 'Highly Rated'])
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], horizontalalignment="center", 
                color="white" if cm[i, j] > cm.max() / 2. else "black")
    st.pyplot(fig)

def main():
    logistic_regression()

if __name__ == "__main__":
    main()

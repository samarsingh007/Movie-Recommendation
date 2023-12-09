# knn.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def knn_regressor():
    st.title("KNN Regressor for Movie Ratings")

    # File Uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    if uploaded_file is not None:
        movies_df = pd.read_csv(uploaded_file)
    else:
        # Default dataset
        csv_file_path = 'src/Dataset/movies.csv'
        movies_df = pd.read_csv(csv_file_path)

    st.write(movies_df.head())

    # Data Preprocessing
    movies_df['YEAR'] = movies_df['YEAR'].str.extract('(\d{4})').astype(float)
    movies_df['VOTES'] = movies_df['VOTES'].str.replace(',', '').astype(float)

    features = movies_df[['YEAR', 'RunTime', 'VOTES']]
    target = movies_df['RATING']

    imputer = SimpleImputer(strategy='median')
    features_imputed = imputer.fit_transform(features)
    target_imputed = target.fillna(target.median())

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_imputed)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_imputed, test_size=0.2, random_state=42)

    # Applying KNN
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Visualize the results
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    ax.set_title('Actual vs Predicted Ratings')
    ax.set_xlabel('Actual Ratings')
    ax.set_ylabel('Predicted Ratings')
    ax.plot([0, 10], [0, 10], color='red', lw=2, linestyle='--')  # Reference line
    st.pyplot(fig)

    # Output the Mean Squared Error
    st.write(f"Mean Squared Error: {mse}")

def main():
    knn_regressor()

if __name__ == "__main__":
    main()

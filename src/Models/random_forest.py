import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def random_forest():
    st.title("Random Forest Model for Movie Ratings")

    # File Uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    if uploaded_file is not None:
        movies_df = pd.read_csv(uploaded_file)
    else:
        # Default dataset
        file_path = 'src/Dataset/movies.csv'
        movies_df = pd.read_csv(file_path)

    # Preprocessing
    simplified_df = movies_df[['GENRE', 'RATING', 'VOTES', 'RunTime']].dropna(subset=['RATING'])
    simplified_df['VOTES'] = simplified_df['VOTES'].str.replace(',', '').astype(int)

    imputer = SimpleImputer(strategy='median')
    simplified_df[['VOTES', 'RunTime']] = imputer.fit_transform(simplified_df[['VOTES', 'RunTime']])
    simplified_df = simplified_df.dropna(subset=['GENRE'])

    X = simplified_df[['GENRE', 'VOTES', 'RunTime']]
    y = simplified_df['RATING']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('genre', OneHotEncoder(handle_unknown='ignore'), ['GENRE']),
            ('num', StandardScaler(), ['VOTES', 'RunTime'])
        ])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Model
    random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('random_forest', random_forest)
    ])
    rf_pipeline.fit(X_train, y_train)

    # Predictions and Performance Metrics
    y_pred_rf = rf_pipeline.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = rf_pipeline.score(X_test, y_test)

    # Display Metrics
    st.write(f"Mean Squared Error: {mse_rf}")
    st.write(f"Mean Absolute Error: {mae_rf}")
    st.write(f"R^2 Score: {r2_rf}")

    # Feature Importances Visualization
    feature_importances = rf_pipeline.named_steps['random_forest'].feature_importances_
    encoded_feature_names = preprocessor.named_transformers_['genre'].get_feature_names_out(input_features=['GENRE'])
    feature_names = np.concatenate((encoded_feature_names, ['VOTES', 'RunTime']))
    importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    top_features = importances_df.sort_values(by='Importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features['Feature'], top_features['Importance'])
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Features')
    ax.set_title('Top 10 Feature Importances in Random Forest Model')
    ax.invert_yaxis()
    st.pyplot(fig)

def main():
    random_forest()

if __name__ == "__main__":
    main()

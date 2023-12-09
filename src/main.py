import streamlit as st
from Models.knn import knn_regressor
from Models.decision_tree import decision_tree_main
from Models.kmeans import kmeans_clustering
from Models.logistic_reg import logistic_regression
from Models.naive_bayes import naive_bayes
from Models.random_forest import random_forest
# Import other models similarly if they are in different files

def main():
    st.title("Machine Learning Model Selection")

    model_options = ['Select a model', 'Decision Tree', 'K-means Clustering', 'KNN Regression', 'Logistic Regression', 'Naive Bayes', 'Random Forest']
    selected_model = st.selectbox("Choose a model to run:", model_options)

    if selected_model == 'Decision Tree':
        decision_tree_main()
    elif selected_model == 'K-means Clustering':
        kmeans_clustering()
    elif selected_model == 'KNN Regression':
        knn_regressor()
    elif selected_model == 'Logistic Regression':
        logistic_regression()
    elif selected_model == 'Naive Bayes':
        naive_bayes_model()
    elif selected_model == 'Random Forest':
        random_forest()
    else:
        st.write("Select a model to display its output.")

if __name__ == "__main__":
    main()
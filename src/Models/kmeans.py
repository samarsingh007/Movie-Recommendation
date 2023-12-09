import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def kmeans_clustering():
    st.title("K-means Clustering for Movie Data")

    # Default dataset path for Excel file
    default_dataset_path = 'src/Dataset/Movie_Gross_Value.xlsx'
    data = pd.read_excel(default_dataset_path)

    # Option to upload a different dataset
    st.subheader("Upload Your Own Dataset (Optional)")
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)

    st.write(data.head())

    # Number of clusters input
    k = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=6)

    X = data[['RATING', 'Gross', 'VOTES']]

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_
    data['Cluster'] = labels

    # 3D Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data['RATING'], data['Gross'], data['VOTES'], c=labels, cmap='rainbow')
    ax.set_xlabel('RATING')
    ax.set_ylabel('Gross')
    ax.set_zlabel('VOTES')
    ax.set_title('K-means Clustering 3D Visualization')
    st.pyplot(fig)

    # 2D Visualization: Rating vs Gross
    fig, ax = plt.subplots()
    scatter = ax.scatter(data['RATING'], data['Gross'], c=labels, cmap='rainbow')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Gross')
    ax.set_title('K-means Clustering: Rating vs Gross')
    st.pyplot(fig)

    # 2D Visualization: Gross vs Votes
    fig, ax = plt.subplots()
    scatter = ax.scatter(data['Gross'], data['VOTES'], c=labels, cmap='rainbow')
    ax.set_xlabel('Gross')
    ax.set_ylabel('VOTES')
    ax.set_title('K-means Clustering: VOTES vs Gross')
    st.pyplot(fig)

    # 2D Visualization: Votes vs Rating
    fig, ax = plt.subplots()
    scatter = ax.scatter(data['VOTES'], data['RATING'], c=labels, cmap='rainbow')
    ax.set_xlabel('VOTES')
    ax.set_ylabel('RATING')
    ax.set_title('K-means Clustering: VOTES vs RATING')
    st.pyplot(fig)

def main():
    kmeans_clustering()

if __name__ == "__main__":
    main()
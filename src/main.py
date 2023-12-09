import streamlit as st
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

# Base directory for files
base_directory = './src/'  # Update this path

# Function to display and execute a Jupyter notebook
def display_notebook(notebook_path):
    if os.path.exists(notebook_path):
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=600)
            ep.preprocess(nb)
            exporter = HTMLExporter()
            body, _ = exporter.from_notebook_node(nb)
            st.markdown(body, unsafe_allow_html=True)
    else:
        st.error(f"File {notebook_path} not found.")

# Function to display and execute a Python script
def display_script(script_path):
    if os.path.exists(script_path):
        with st.echo():
            exec(open(script_path).read(), globals())
    else:
        st.error(f"File {script_path} not found.")

# Streamlit application
def main():
    st.title("Machine Learning Models Display")

    st.sidebar.title("Select a File to Display")
    options = [os.path.join(base_directory, filename) for filename in 
               ["Decision_Tree.ipynb", "kmeans.py", "KNN.ipynb", "Logistic_Regression.ipynb", "nb.py", "RF.ipynb"]]
    selected_file = st.sidebar.selectbox("Choose a file", options)

    if selected_file.endswith('.ipynb'):
        display_notebook(selected_file)
    else:
        display_script(selected_file)

if __name__ == "__main__":
    main()

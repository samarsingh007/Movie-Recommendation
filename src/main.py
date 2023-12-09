import streamlit as st
import os
import subprocess

# Function to display a Jupyter notebook
def display_notebook(notebook_path):
    if os.path.exists(notebook_path):
        subprocess.run(['jupyter', 'nbconvert', '--to', 'html', notebook_path])
        html_file = notebook_path.replace('.ipynb', '.html')
        if os.path.exists(html_file):
            with open(html_file, 'r') as f:
                st.markdown(f.read(), unsafe_allow_html=True)
        else:
            st.error(f"Failed to convert {notebook_path} to HTML.")
    else:
        st.error(f"File {notebook_path} not found.")

# Function to display a Python script
def display_script(script_path):
    if os.path.exists(script_path):
        with open(script_path, 'r') as f:
            code = f.read()
            st.code(code, language='python')
    else:
        st.error(f"File {script_path} not found.")

# Streamlit application
def main():
    st.title("Machine Learning Models Display")

    st.sidebar.title("Select a File to Display")
    options = ["Decision_Tree.ipynb", "kmeans.py", "KNN.ipynb", "Logistic_Regression.ipynb", "nb.py", "RF.ipynb"]
    selected_file = st.sidebar.selectbox("Choose a file", options)

    if selected_file.endswith('.ipynb'):
        display_notebook(selected_file)
    else:
        display_script(selected_file)

if __name__ == "__main__":
    main()

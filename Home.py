import streamlit as st

st.set_page_config(
    page_title="Dataset Generator",
    page_icon="📊",  # Updated favicon to degree hat
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Dataset Generator")
# st.sidebar.success("Select a generator from the menu above.")

st.markdown("""
## Welcome to Dataset Generator!

This application helps you generate various types of datasets for machine learning tasks. Choose from the following options in the sidebar:

### 1. Classification Datasets 📋
- Generate standard classification datasets
- Multiple patterns: Linear, Circular, Moons, etc.
- Control number of classes and noise levels

### 2. Regression Datasets 📈
- Generate standard regression datasets
- Various functions: Linear, Quadratic, Cubic, etc.
- Control parameters and add noise

### 3. Custom Classification Dataset ✏️
- Draw your own classification patterns
- Interactive canvas with spray effect
- Multiple classes with different colors

### 4. Custom Regression Dataset 🎨
- Draw custom regression patterns
- Interactive line drawing
- Control spray effect and density

Choose a generator from the sidebar to get started!
""")

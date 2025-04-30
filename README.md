## Dataset Generator Streamlit App

This simple web-based UI allows you to generate classification and regression datasets. You can generate different types of datasets by adjusting some parameters.

## Access online app
You can access online streamlit app using the following URL without downloading or installing anything.
[https://generatedataset.streamlit.app/](https://generatedataset.streamlit.app/)

## App Features

The Dataset Generator Streamlit App provides the following functionalities:

### 1. Classification Dataset Generator
- Generate standard classification datasets with patterns such as Linear, Circular, Moons, etc.
- Customize parameters like the number of samples, noise level, and number of classes.
- Apply transformations such as flipping axes and shifting data.

### 2. Regression Dataset Generator
- Generate regression datasets using functions like Linear, Quadratic, Cubic, Sinusoidal, etc.
- Adjust parameters such as slope, intercept, noise, and more.
- Apply transformations to the generated data.

### 3. Custom Classification Dataset Generator
- Draw custom classification patterns using an interactive canvas.
- Use a spray effect to create points for multiple classes with different colors.
- Customize transformations and download the generated dataset.

### 4. Custom Regression Dataset Generator
- Draw freehand regression patterns on an interactive canvas.
- Generate points along the drawn path with a spray effect.
- Customize transformations and download the generated dataset.

## Setup
```bash 
git clone https://github.com/bit-magpie/Classification_learner_streamlit.git
cd Classification_learner_streamlit
python -m venv dgenv
source ./dgenv/bin/activate
pip install -r requirements.txt
```

## Execute program
```bash
streamlit run app.py
```
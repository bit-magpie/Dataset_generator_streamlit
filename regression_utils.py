import numpy as np
from sklearn.datasets import make_regression
import plotly.express as px
import pandas as pd

# Parameters for axis transformations
AXIS_PARAMS = {
    "flip_x": {"default": False},
    "flip_y": {"default": False},
    "shift_x": {"min": -5.0, "max": 5.0, "default": 0.0, "step": 0.5},
    "shift_y": {"min": -5.0, "max": 5.0, "default": 0.0, "step": 0.5}
}

def apply_transformations(X, y, flip_x=False, flip_y=False, shift_x=0.0, shift_y=0.0):
    """Apply axis transformations to the dataset"""
    if flip_x:
        X = -X
    if flip_y:
        y = -y
    X = X + shift_x
    y = y + shift_y
    return X, y

def generate_linear_dataset(n_samples, noise, slope=1.0, intercept=0.0, 
                          flip_x=False, flip_y=False, shift_x=0.0, shift_y=0.0):
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = slope * X[:, 0] + intercept + np.random.normal(0, noise, n_samples)
    return apply_transformations(X, y, flip_x, flip_y, shift_x, shift_y)

def generate_quadratic_dataset(n_samples, noise, a=0.5, b=1.0, c=0.0,
                             flip_x=False, flip_y=False, shift_x=0.0, shift_y=0.0):
    """Quadratic function: y = ax² + bx + c"""
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = a * X[:, 0]**2 + b * X[:, 0] + c + np.random.normal(0, noise, n_samples)
    return apply_transformations(X, y, flip_x, flip_y, shift_x, shift_y)

def generate_cubic_dataset(n_samples, noise, a=0.1, b=0.5, c=-2.0, d=0.0,
                         flip_x=False, flip_y=False, shift_x=0.0, shift_y=0.0):
    """Cubic function: y = ax³ + bx² + cx + d"""
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = (a * X[:, 0]**3 + b * X[:, 0]**2 + c * X[:, 0] + d + 
         np.random.normal(0, noise, n_samples))
    return apply_transformations(X, y, flip_x, flip_y, shift_x, shift_y)

def generate_sinusoidal_dataset(n_samples, noise, amplitude=1.0, frequency=1.0, phase=0.0,
                              flip_x=False, flip_y=False, shift_x=0.0, shift_y=0.0):
    """Sinusoidal function: y = A * sin(fx + p)"""
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = amplitude * np.sin(frequency * X[:, 0] + phase) + np.random.normal(0, noise, n_samples)
    return apply_transformations(X, y, flip_x, flip_y, shift_x, shift_y)

def generate_radial_dataset(n_samples, noise, scale=1.0, exponent=0.5,
                          flip_x=False, flip_y=False, shift_x=0.0, shift_y=0.0):
    """Radial function: y = scale * x^exponent"""
    X = np.linspace(0, 5, n_samples).reshape(-1, 1)
    y = scale * np.power(X[:, 0], exponent) + np.random.normal(0, noise, n_samples)
    return apply_transformations(X, y, flip_x, flip_y, shift_x, shift_y)

def generate_exponential_dataset(n_samples, noise, base=np.e, scale=1.0, shift=0.0,
                               flip_x=False, flip_y=False, shift_x=0.0, shift_y=0.0):
    """Exponential function: y = scale * base^x + shift"""
    X = np.linspace(0, 3, n_samples).reshape(-1, 1)
    y = scale * np.power(base, X[:, 0]) + shift + np.random.normal(0, noise, n_samples)
    return apply_transformations(X, y, flip_x, flip_y, shift_x, shift_y)

def plot_regression_dataset(X, y, title="Regression Dataset"):
    df = pd.DataFrame({
        'Feature': X[:, 0],
        'Target': y
    })
    
    fig = px.scatter(
        df, x='Feature', y='Target',
        title=title,
        template='plotly_white',
        labels={'Feature': 'X', 'Target': 'y'}
    )
    
    return fig, df

DATASET_GENERATORS = {
    "Linear": generate_linear_dataset,
    "Quadratic": generate_quadratic_dataset,
    "Cubic": generate_cubic_dataset,
    "Sinusoidal": generate_sinusoidal_dataset,
    "Radial": generate_radial_dataset,
    "Exponential": generate_exponential_dataset
}

# Parameters for each regression type
REGRESSION_PARAMS = {
    "Linear": {
        "slope": {"min": -3.0, "max": 3.0, "default": 1.0, "step": 0.1},
        "intercept": {"min": -3.0, "max": 3.0, "default": 0.0, "step": 0.1}
    },
    "Quadratic": {
        "a": {"min": -1.0, "max": 1.0, "default": 0.5, "step": 0.1},
        "b": {"min": -3.0, "max": 3.0, "default": 1.0, "step": 0.1},
        "c": {"min": -3.0, "max": 3.0, "default": 0.0, "step": 0.1}
    },
    "Cubic": {
        "a": {"min": -0.5, "max": 0.5, "default": 0.1, "step": 0.05},
        "b": {"min": -1.0, "max": 1.0, "default": 0.5, "step": 0.1},
        "c": {"min": -3.0, "max": 3.0, "default": -2.0, "step": 0.1},
        "d": {"min": -3.0, "max": 3.0, "default": 0.0, "step": 0.1}
    },
    "Sinusoidal": {
        "amplitude": {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1},
        "frequency": {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1},
        "phase": {"min": -3.14, "max": 3.14, "default": 0.0, "step": 0.1}
    },
    "Radial": {
        "scale": {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1},
        "exponent": {"min": 0.1, "max": 3.0, "default": 0.5, "step": 0.1}
    },
    "Exponential": {
        "scale": {"min": 0.1, "max": 2.0, "default": 1.0, "step": 0.1},
        "base": {"min": 1.1, "max": 4.0, "default": np.e, "step": 0.1},
        "shift": {"min": -3.0, "max": 3.0, "default": 0.0, "step": 0.1}
    }
}
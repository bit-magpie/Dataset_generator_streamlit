import numpy as np
from sklearn.datasets import make_classification, make_circles, make_moons
import plotly.express as px
import pandas as pd

# Parameters for each dataset type
CLASSIFICATION_PARAMS = {
    "Linear": {
        "class_sep": {"min": 0.5, "max": 3.0, "default": 1.0, "step": 0.1},
    },
    "Circular": {
        "factor": {"min": 0.1, "max": 0.9, "default": 0.5, "step": 0.05},
    },
    "Moons": {
        "shift": {"min": -1.0, "max": 1.0, "default": 0.0, "step": 0.1},
    },
    "Quadratic": {
        "threshold": {"min": 0.5, "max": 2.0, "default": 1.0, "step": 0.1},
    },
    "Cubic": {
        "threshold": {"min": -1.0, "max": 1.0, "default": 0.0, "step": 0.1},
    },
    "Radial": {
        "radius_bins": {"min": 2, "max": 5, "default": 3, "step": 1},
    }
}

# Transformation parameters
AXIS_PARAMS = {
    "flip_x": {"default": False},
    "flip_y": {"default": False},
    "shift_x": {"min": -5.0, "max": 5.0, "default": 0.0, "step": 0.5},
    "shift_y": {"min": -5.0, "max": 5.0, "default": 0.0, "step": 0.5},
}

def apply_transformations(X, flip_x=False, flip_y=False, shift_x=0.0, shift_y=0.0):
    """Apply axis transformations to the dataset"""
    X_transformed = X.copy()
    
    if flip_x:
        X_transformed[:, 0] = -X_transformed[:, 0]
    if flip_y:
        X_transformed[:, 1] = -X_transformed[:, 1]
    
    X_transformed[:, 0] = X_transformed[:, 0] + shift_x
    X_transformed[:, 1] = X_transformed[:, 1] + shift_y
    
    return X_transformed

def generate_linear_dataset(n_samples, n_classes, noise=0.1, class_sep=1.0, **kwargs):
    """Generate a random n-class classification dataset with optional noise"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        n_classes=n_classes,
        random_state=42,
        class_sep=class_sep,
        flip_y=noise  # Using flip_y as our noise parameter
    )
    # Add gaussian noise to the features
    X = X + np.random.normal(0, noise, X.shape)
    
    # Apply transformations
    X = apply_transformations(
        X, 
        kwargs.get('flip_x', False),
        kwargs.get('flip_y', False),
        kwargs.get('shift_x', 0.0),
        kwargs.get('shift_y', 0.0)
    )
    
    return X, y

def generate_circular_dataset(n_samples, n_classes=None, noise=0.1, factor=0.5, **kwargs):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
    
    # Apply transformations
    X = apply_transformations(
        X, 
        kwargs.get('flip_x', False),
        kwargs.get('flip_y', False),
        kwargs.get('shift_x', 0.0),
        kwargs.get('shift_y', 0.0)
    )
    
    return X, y

def generate_moons_dataset(n_samples, n_classes=None, noise=0.1, shift=0.0, **kwargs):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    
    # Apply custom shift parameter
    X[:, 1] = X[:, 1] + shift
    
    # Apply transformations
    X = apply_transformations(
        X, 
        kwargs.get('flip_x', False),
        kwargs.get('flip_y', False),
        kwargs.get('shift_x', 0.0),
        kwargs.get('shift_y', 0.0)
    )
    
    return X, y

def generate_quadratic_dataset(n_samples, n_classes=None, noise=0.1, threshold=1.0, **kwargs):
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0]**2 + X[:, 1]**2) > threshold).astype(int)
    X = X + np.random.normal(0, noise, X.shape)
    
    # Apply transformations
    X = apply_transformations(
        X, 
        kwargs.get('flip_x', False),
        kwargs.get('flip_y', False),
        kwargs.get('shift_x', 0.0),
        kwargs.get('shift_y', 0.0)
    )
    
    return X, y

def generate_cubic_dataset(n_samples, n_classes=None, noise=0.1, threshold=0.0, **kwargs):
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0]**3 + X[:, 1]**3) > threshold).astype(int)
    X = X + np.random.normal(0, noise, X.shape)
    
    # Apply transformations
    X = apply_transformations(
        X, 
        kwargs.get('flip_x', False),
        kwargs.get('flip_y', False),
        kwargs.get('shift_x', 0.0),
        kwargs.get('shift_y', 0.0)
    )
    
    return X, y

def generate_radial_dataset(n_samples, n_classes=3, noise=0.1, radius_bins=3, **kwargs):
    np.random.seed(42)
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    r = np.random.uniform(0, 3, n_samples)
    X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    y = (r / 3 * radius_bins).astype(int)
    y = np.clip(y, 0, radius_bins-1)
    X = X + np.random.normal(0, noise, X.shape)
    
    # Apply transformations
    X = apply_transformations(
        X, 
        kwargs.get('flip_x', False),
        kwargs.get('flip_y', False),
        kwargs.get('shift_x', 0.0),
        kwargs.get('shift_y', 0.0)
    )
    
    return X, y

def plot_classification_dataset(X, y, title="Classification Dataset"):
    df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    df['Target'] = y
    
    fig = px.scatter(
        df, x='Feature 1', y='Feature 2',
        color='Target',
        title=title,
        template='plotly_white',
        color_continuous_scale='viridis'
    )
    
    return fig, df

DATASET_GENERATORS = {
    "Linear": generate_linear_dataset,
    "Circular": generate_circular_dataset,
    "Moons": generate_moons_dataset,
    "Quadratic": generate_quadratic_dataset,
    "Cubic": generate_cubic_dataset,
    "Radial": generate_radial_dataset
}
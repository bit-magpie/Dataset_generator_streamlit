import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Default color palette in hex format
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def create_spray_points(x, y, radius, n_points=20):
    """Create spray effect points around a center point"""
    angles = np.random.uniform(0, 2*np.pi, n_points)
    distances = np.random.normal(0, radius/2, n_points)
    points_x = x + distances * np.cos(angles)
    points_y = y + distances * np.sin(angles)
    return points_x, points_y

def apply_transformations(X, flip_x=False, flip_y=False, shift_x=0.0, shift_y=0.0):
    """Apply axis transformations to the dataset"""
    # Convert to float arrays to ensure proper operations
    X = X.astype(float)
    
    if flip_x:
        X[:, 0] = -X[:, 0]
    if flip_y:
        X[:, 1] = -X[:, 1]
    
    X[:, 0] = X[:, 0] + shift_x
    X[:, 1] = X[:, 1] + shift_y
    
    return X

def draw_classification_canvas():
    """Create an interactive canvas for drawing classification data"""
    # Create columns for spray parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        spray_radius = st.slider("Spray Radius", 5, 50, 20)
    with col2:
        points_per_spray = st.slider("Points per Spray", 10, 100, 30)
    with col3:
        n_classes = st.number_input("Number of Classes", 2, 10, 3)
    
    # Create color pickers in a row
    color_cols = st.columns(n_classes)
    colors = []
    for i, col in enumerate(color_cols):
        with col:
            color = st.color_picker(f"Class {i+1}", 
                                value=DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            colors.append(color)
    
    # Class selection in its own row
    current_color = st.selectbox("Current Drawing Class", 
                             range(n_classes), 
                             format_func=lambda x: f"Class {x+1}")
    
    # Transformation controls row
    st.markdown("### Plot Transformations")
    trans_col1, trans_col2, trans_col3, trans_col4 = st.columns(4)
    with trans_col1:
        flip_x = st.checkbox("Flip X Axis", value=False)
    with trans_col2:
        flip_y = st.checkbox("Flip Y Axis", value=False)  # Changed to False as Y-axis is automatically flipped
    with trans_col3:
        shift_x = st.number_input("Shift X", min_value=-100.0, max_value=100.0, value=0.0, step=10.0)
    with trans_col4:
        shift_y = st.number_input("Shift Y", min_value=-100.0, max_value=100.0, value=0.0, step=10.0)
    
    return {
        "stroke_width": 10,  # Fixed width
        "stroke_color": colors[current_color],
        "spray_radius": spray_radius,
        "points_per_spray": points_per_spray,
        "n_classes": n_classes,
        "colors": colors,
        "flip_x": flip_x,
        "flip_y": flip_y,
        "shift_x": shift_x,
        "shift_y": shift_y
    }

def process_canvas_data(canvas_result, params):
    """Process canvas data and convert to dataset"""
    if canvas_result.json_data is None or not canvas_result.json_data["objects"]:
        return None, None
    
    all_points_x = []
    all_points_y = []
    all_classes = []
    
    # Get canvas height
    canvas_height = canvas_result.json_data.get("height", 400)
    
    # Create a mapping of colors to class labels
    color_to_class = {color: f"class{i+1}" for i, color in enumerate(params["colors"])}
    
    for obj in canvas_result.json_data["objects"]:
        if "type" in obj and obj["type"] == "circle":
            x = float(obj["left"]) + float(obj["radius"])
            # Invert Y coordinate to match mathematical coordinate system (y increases upward)
            y = canvas_height - (float(obj["top"]) + float(obj["radius"]))
            color = obj.get("stroke", "#000000")  # Get color
            class_label = color_to_class.get(color, "unknown")  # Convert color to class label
            
            # Create spray effect points
            points_x, points_y = create_spray_points(x, y, params["spray_radius"], params["points_per_spray"])
            
            # Convert numpy arrays to lists for JSON serialization
            all_points_x.extend(points_x.tolist())
            all_points_y.extend(points_y.tolist())
            all_classes.extend([class_label] * len(points_x))
    
    if not all_points_x:
        return None, None
        
    X = np.column_stack([all_points_x, all_points_y])
    y = np.array(all_classes)
    
    return X, y

def plot_custom_classification(X, y, params=None, title="Custom Classification Dataset"):
    """Plot the custom classification dataset"""
    if params:
        X = apply_transformations(X, 
                                params["flip_x"], 
                                params["flip_y"], 
                                params["shift_x"], 
                                params["shift_y"])
    
    df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    df['Target'] = y
    
    # Create a color mapping for the plot
    color_discrete_map = {f"class{i+1}": color for i, color in enumerate(params["colors"])}
    
    fig = px.scatter(
        df, x='Feature 1', y='Feature 2',
        color='Target',
        title=title,
        template='plotly_white',
        color_discrete_map=color_discrete_map
    )
    
    # Customize the plot layout
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(
        showlegend=True,
        plot_bgcolor='white'
    )
    
    return fig, df
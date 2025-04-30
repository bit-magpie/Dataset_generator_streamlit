import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_drawable_canvas import st_canvas

def create_spray_points_along_path(path_points, spray_radius, points_per_segment=20):
    """Create spray effect points along a path"""
    all_points_x = []
    all_points_y = []
    
    for i in range(len(path_points) - 1):
        x1, y1 = path_points[i]
        x2, y2 = path_points[i + 1]
        
        # Create points along the line segment
        t = np.linspace(0, 1, points_per_segment)
        base_x = x1 + (x2 - x1) * t
        base_y = y1 + (y2 - y1) * t
        
        # Add random noise to create spray effect
        for x, y in zip(base_x, base_y):
            noise_x = np.random.normal(0, spray_radius, 1)
            noise_y = np.random.normal(0, spray_radius, 1)
            all_points_x.append(float(x + noise_x[0]))
            all_points_y.append(float(y + noise_y[0]))
    
    return np.array(all_points_x), np.array(all_points_y)

def apply_transformations(X, y, flip_x=False, flip_y=False, shift_x=0.0, shift_y=0.0):
    """Apply axis transformations to the dataset"""
    # Convert to float arrays to ensure proper operations
    X = X.astype(float)
    y = y.astype(float)
    
    if flip_x:
        X = -X
    if flip_y:
        y = -y
    
    X = X + shift_x
    y = y + shift_y
    
    return X, y

def process_regression_canvas(canvas_result, spray_radius, points_per_segment):
    """Process canvas data and convert to regression dataset"""
    if canvas_result.json_data is None:
        return None, None
    
    path_points = []
    canvas_height = canvas_result.json_data.get("height", 400)  # Get canvas height
    
    # Handle freehand drawing paths
    if "objects" in canvas_result.json_data:
        for obj in canvas_result.json_data["objects"]:
            if "path" in obj:
                path_data = obj["path"]
                
                for cmd in path_data:
                    if len(cmd) >= 3:  # Make sure we have at least x,y coordinates
                        command = cmd[0]
                        if command == "M":  # Move to
                            x, y = float(cmd[1]), float(cmd[2])
                            # Invert Y coordinate to match mathematical coordinate system (y increases upward)
                            y = canvas_height - y
                            path_points.append((x, y))
                        elif command == "Q":  # Quadratic curve
                            # For Q command, we get both control point and end point
                            if len(cmd) >= 5:
                                cp_x, cp_y = float(cmd[1]), float(cmd[2])
                                end_x, end_y = float(cmd[3]), float(cmd[4])
                                # Invert Y coordinates
                                cp_y = canvas_height - cp_y
                                end_y = canvas_height - end_y
                                path_points.append((cp_x, cp_y))
                                path_points.append((end_x, end_y))
    
    if len(path_points) < 2:
        return None, None
    
    # Generate spray points along the path
    X, y = create_spray_points_along_path(path_points, spray_radius, points_per_segment)
    
    # Sort points by x coordinate for regression
    sort_idx = np.argsort(X)
    X = X[sort_idx]
    y = y[sort_idx]
    
    # We've already inverted Y coordinates from canvas, so no need to flip again
    return X.reshape(-1, 1), y

def plot_custom_regression(X, y, params=None, title="Custom Regression Dataset"):
    """Plot the custom regression dataset"""
    if params:
        X, y = apply_transformations(X, y, 
                                   params["flip_x"], 
                                   params["flip_y"], 
                                   params["shift_x"], 
                                   params["shift_y"])
    
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
    
    # Customize the plot layout
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white'
    )
    
    return fig, df

def draw_regression_canvas():
    """Create an interactive canvas for drawing regression data"""
    # Drawing parameters row
    col1, col2 = st.columns(2)
    with col1:
        spray_radius = st.slider("Spray Radius", 1, 30, 10)
    with col2:
        points_per_segment = st.slider("Points per Segment", 10, 100, 30)
    
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
        "stroke_width": 3,
        "spray_radius": spray_radius,
        "points_per_segment": points_per_segment,
        "flip_x": flip_x,
        "flip_y": flip_y,
        "shift_x": shift_x,
        "shift_y": shift_y
    }
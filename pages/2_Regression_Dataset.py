import streamlit as st
import regression_utils as rg

st.set_page_config(page_title="Regression Dataset Generator", layout="wide")
st.title("Regression Dataset Generator")

# Create two columns for control panel and visualization
control_col, viz_col = st.columns([1, 2])

with control_col:
    st.subheader("Control Panel")
    regression_type = st.selectbox(
        "Select Regression Type",
        list(rg.DATASET_GENERATORS.keys())
    )
    
    reg_n_samples = st.number_input(
        "Number of Samples",
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
        key="reg_samples"
    )
    
    reg_noise = st.slider(
        "Noise Level",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        key="reg_noise"
    )

    st.subheader("Axis Transformations")
    col1, col2 = st.columns(2)
    with col1:
        flip_x = st.checkbox("Flip X Axis", value=rg.AXIS_PARAMS["flip_x"]["default"])
        shift_x = st.number_input(
            "Shift X Axis",
            min_value=float(rg.AXIS_PARAMS["shift_x"]["min"]),
            max_value=float(rg.AXIS_PARAMS["shift_x"]["max"]),
            value=float(rg.AXIS_PARAMS["shift_x"]["default"]),
            step=float(rg.AXIS_PARAMS["shift_x"]["step"]),
            key="shift_x"
        )
    with col2:
        flip_y = st.checkbox("Flip Y Axis", value=rg.AXIS_PARAMS["flip_y"]["default"])
        shift_y = st.number_input(
            "Shift Y Axis",
            min_value=float(rg.AXIS_PARAMS["shift_y"]["min"]),
            max_value=float(rg.AXIS_PARAMS["shift_y"]["max"]),
            value=float(rg.AXIS_PARAMS["shift_y"]["default"]),
            step=float(rg.AXIS_PARAMS["shift_y"]["step"]),
            key="shift_y"
        )

    st.subheader("Function Parameters")
    # Get parameters for the selected regression type
    params = rg.REGRESSION_PARAMS[regression_type]
    param_values = {}
    
    # Create sliders for each parameter
    for param_name, param_config in params.items():
        param_values[param_name] = st.slider(
            f"{param_name.capitalize()}",
            min_value=float(param_config["min"]),
            max_value=float(param_config["max"]),
            value=float(param_config["default"]),
            step=float(param_config["step"]),
            key=f"reg_{regression_type}_{param_name}"
        )
    
    # Add axis transformation parameters to the generator call
    param_values.update({
        "flip_x": flip_x,
        "flip_y": flip_y,
        "shift_x": shift_x,
        "shift_y": shift_y
    })

with viz_col:
    # Generate dataset with current parameters
    generator = rg.DATASET_GENERATORS[regression_type]
    X, y = generator(reg_n_samples, reg_noise, **param_values)
    
    fig, df = rg.plot_regression_dataset(X, y, f"{regression_type} Regression Dataset")
    
    # Place download button above the plot
    st.download_button(
        label="Download Dataset as CSV",
        data=df.to_csv(index=False),
        file_name=f"{regression_type.lower()}_dataset.csv",
        mime="text/csv"
    )
    
    st.plotly_chart(fig, use_container_width=True)
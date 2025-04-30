import streamlit as st
import classification_utils as cg

st.set_page_config(page_title="Classification Dataset Generator", layout="wide")
st.title("Classification Dataset Generator")

# Create two columns for control panel and visualization
control_col, viz_col = st.columns([1, 2])

with control_col:
    st.subheader("Control Panel")
    classification_type = st.selectbox(
        "Select Classification Type",
        list(cg.DATASET_GENERATORS.keys())
    )
    
    st.markdown("### Basic Parameters")
    n_samples = st.number_input(
        "Number of Samples",
        min_value=50,
        max_value=1000,
        value=200,
        step=50
    )
    
    noise = st.slider(
        "Noise Level",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05
    )
    
    n_classes = st.number_input(
        "Number of Classes",
        min_value=2,
        max_value=5,
        value=2,
        step=1,
        disabled=classification_type in ["Circular", "Moons"]
    )
    
    # Dataset-specific parameters
    st.markdown("### Dataset Parameters")
    dataset_params = {}
    
    if classification_type in cg.CLASSIFICATION_PARAMS:
        param_config = cg.CLASSIFICATION_PARAMS[classification_type]
        
        for param_name, param_settings in param_config.items():
            if param_name == "radius_bins" and classification_type == "Radial":
                dataset_params[param_name] = st.number_input(
                    f"{param_name.replace('_', ' ').title()}",
                    min_value=param_settings["min"],
                    max_value=param_settings["max"],
                    value=n_classes if n_classes >= 2 and n_classes <= 5 else param_settings["default"],
                    step=param_settings["step"]
                )
            else:
                dataset_params[param_name] = st.slider(
                    f"{param_name.replace('_', ' ').title()}",
                    min_value=param_settings["min"],
                    max_value=param_settings["max"],
                    value=param_settings["default"],
                    step=param_settings["step"]
                )
    
    # Transformation parameters
    st.markdown("### Plot Transformations")
    trans_col1, trans_col2 = st.columns(2)
    with trans_col1:
        flip_x = st.checkbox("Flip X Axis", value=cg.AXIS_PARAMS["flip_x"]["default"])
        shift_x = st.number_input(
            "Shift X", 
            min_value=cg.AXIS_PARAMS["shift_x"]["min"],
            max_value=cg.AXIS_PARAMS["shift_x"]["max"],
            value=cg.AXIS_PARAMS["shift_x"]["default"],
            step=cg.AXIS_PARAMS["shift_x"]["step"]
        )
    with trans_col2:
        flip_y = st.checkbox("Flip Y Axis", value=cg.AXIS_PARAMS["flip_y"]["default"])
        shift_y = st.number_input(
            "Shift Y", 
            min_value=cg.AXIS_PARAMS["shift_y"]["min"],
            max_value=cg.AXIS_PARAMS["shift_y"]["max"],
            value=cg.AXIS_PARAMS["shift_y"]["default"],
            step=cg.AXIS_PARAMS["shift_y"]["step"]
        )
    
    # Add transformations to parameters
    dataset_params.update({
        'flip_x': flip_x,
        'flip_y': flip_y,
        'shift_x': shift_x,
        'shift_y': shift_y
    })

with viz_col:
    # Generate dataset based on current parameters
    generator = cg.DATASET_GENERATORS[classification_type]
    if classification_type in ["Linear", "Radial"]:
        if classification_type == "Radial" and "radius_bins" in dataset_params:
            # For Radial dataset, use the radius_bins parameter for class count
            X, y = generator(n_samples, dataset_params["radius_bins"], noise, **dataset_params)
        else:
            X, y = generator(n_samples, n_classes, noise, **dataset_params)
    else:
        X, y = generator(n_samples, None, noise, **dataset_params)
    
    fig, df = cg.plot_classification_dataset(X, y, f"{classification_type} Classification Dataset")
    
    # Place download button above the plot
    st.download_button(
        label="Download Dataset as CSV",
        data=df.to_csv(index=False),
        file_name=f"{classification_type.lower()}_dataset.csv",
        mime="text/csv"
    )
    
    st.plotly_chart(fig, use_container_width=True)
import streamlit as st
import custom_regression_utils as crg
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Custom Regression Dataset Generator", layout="wide")
st.title("Custom Regression Dataset Generator")
st.write("Draw a freehand curve on the canvas. Points will be generated along the path with spray effect.")

# Get drawing parameters at the top
canvas_params = crg.draw_regression_canvas()

# Add a clear button
# if st.button("Clear Canvas"):
#     st.session_state["regression_canvas"] = None

# Create two columns for canvas and plot
canvas_col, plot_col = st.columns(2)

with canvas_col:
    st.markdown("### Drawing Canvas")
    st.markdown("👇 **Draw a freehand curve below**")
    
    # Create canvas with container
    canvas_result = st_canvas(
        stroke_width=canvas_params["stroke_width"],
        stroke_color="#000000",
        background_color="#fff",
        width=600,  # Adjusted width for side-by-side layout
        height=400,
        drawing_mode="freedraw",
        key="regression_canvas",
    )

with plot_col:
    st.markdown("### Generated Dataset")
    if canvas_result is not None and canvas_result.json_data is not None:
        X, y = crg.process_regression_canvas(
            canvas_result, 
            canvas_params["spray_radius"], 
            canvas_params["points_per_segment"]
        )
        if X is not None and y is not None:
            fig, df = crg.plot_custom_regression(X, y, params=canvas_params)
            
            # Place download button above the plot
            st.download_button(
                label="Download Dataset as CSV",
                data=df.to_csv(index=False),
                file_name="custom_regression_dataset.csv",
                mime="text/csv"
            )
            
            st.plotly_chart(fig, use_container_width=True)
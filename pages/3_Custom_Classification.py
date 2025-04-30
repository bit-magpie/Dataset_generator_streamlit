import streamlit as st
import custom_classification_utils as ccg
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Custom Classification Dataset Generator", layout="wide")
st.title("Custom Classification Dataset Generator")
st.write("Draw points on the canvas using spray effect. Each class can have a different color.")

# Get drawing parameters at the top
canvas_params = ccg.draw_classification_canvas()

# Add a clear button
# if st.button("Clear Canvas"):
#     st.session_state["classification_canvas"] = None

# Create two columns for canvas and plot
canvas_col, plot_col = st.columns(2)

with canvas_col:
    st.markdown("### Drawing Canvas")
    st.markdown("👇 **Click to draw points below**")
    
    # Create canvas with container
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=canvas_params["stroke_width"],
        stroke_color=canvas_params["stroke_color"],
        background_color="#fff",
        width=600,  # Adjusted width for side-by-side layout
        height=400,
        drawing_mode="circle",
        key="classification_canvas",
    )

with plot_col:
    st.markdown("### Generated Dataset")
    if canvas_result is not None and canvas_result.json_data is not None:
        X, y = ccg.process_canvas_data(canvas_result, canvas_params)
        if X is not None and y is not None:
            fig, df = ccg.plot_custom_classification(X, y, params=canvas_params)
            
            # Place download button above the plot
            st.download_button(
                label="Download Dataset as CSV",
                data=df.to_csv(index=False),
                file_name="custom_classification_dataset.csv",
                mime="text/csv"
            )
            
            st.plotly_chart(fig, use_container_width=True)
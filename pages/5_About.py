import streamlit as st

_, col, _ = st.columns([3,6,3])
with col:
    st.header("About")
    
    with st.container():
        st.write("#### Dataset Generator V1.0.0")
        st.write("**Author:** Isuru Jayarathne")
        st.write("**Git repo:** [https://github.com/bit-magpie/Dataset_generator_streamlit](https://github.com/bit-magpie/Dataset_generator_streamlit)")
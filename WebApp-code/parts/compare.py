import streamlit as st
from PIL import Image
import io
import copy
from streamlit_image_comparison import image_comparison

# Function for Visualising masks using slider
def compare_images():
    # Buttons for Navigating
    btn1, btn2, _, _ = st.columns(4)
    if btn1.button("Back to Normal View"):
        st.session_state.page = "submitted_uploads"
        st.rerun()
    if btn2.button("Upload New Files"):
        st.session_state.page = "basic_uploads"
        st.session_state.uploaded_files = None
        st.rerun()

    # Image Comparison Using Slider
    _, col2, _ = st.columns([1,4,1])
    uploaded_files_copy = [copy.copy(uploaded_file) for uploaded_file in st.session_state.uploaded_files] 
    segmented_files_copy = [copy.copy(segmented_file) for segmented_file in st.session_state.segmented_files] 
    with col2:
        for i in range(len(uploaded_files_copy)):
            if st.session_state.edited == True:
                input_image = uploaded_files_copy[i]
            else:
                image_data = uploaded_files_copy[i].read()
                input_image = Image.open(io.BytesIO(image_data))
            image_comparison(
                input_image,
                segmented_files_copy[i],
                label1 = "Original",
                label2 = "Masked",
                width = 450
            )
    del uploaded_files_copy
    del segmented_files_copy
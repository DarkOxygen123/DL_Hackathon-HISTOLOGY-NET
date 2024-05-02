import streamlit as st
from PIL import Image
import io

# Function for Images Upload Page
def basic_uploads_page():
    # Uploading images Functionality
    uploaded_files = st.file_uploader("Upload image(s) for segmentation here:", type=["jpg", "png"], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.names = [image.name for image in st.session_state.uploaded_files]
    st.text("")

    # Buttons for navigating
    _, _, btn1, btn2 = st.columns(4)
    if btn1.button("Enhance/Edit Images", help = "Crop/Rotate the input images before segmentation"):
        if st.session_state.uploaded_files is None:
            st.warning("Please select at least one image file to proceed.")
        else:
            st.session_state.page = "enhance_uploads"
            st.rerun()
    if btn2.button("Submit for Segmentation", help = "Directly submit the images to segmentation model"):
        if st.session_state.uploaded_files is None:
            st.warning("Please select at least one image file to proceed.")
        else:
            st.session_state.page = "submitted_uploads"
            st.rerun()

    # Displaying thumbnails of uploaded images
    st.markdown("***")
    st.write("Uploaded Images:")
    cols=st.columns(4)
    if (st.session_state.uploaded_files):
        for i,uploaded_file in enumerate(st.session_state.uploaded_files):
                try:
                    image_data = uploaded_file.read()
                    img = Image.open(io.BytesIO(image_data))
                    cols[i%4].image(img,caption=uploaded_file.name, use_column_width=True)
                    # st.image(img, caption=uploaded_file.name, use_column_width=False)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}:{e}")

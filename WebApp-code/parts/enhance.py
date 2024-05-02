import streamlit as st
import copy
from PIL import Image
import io
from streamlit_cropper import st_cropper

# Function for Enhance uploads Page
def enhance_uploads_page():
    images_uploaded = st.session_state.uploaded_files
    image_enhance(images_uploaded)

    # Buttons for Navigating
    _, _, btn1, btn2 = st.columns(4)
    if btn1.button("Go Back", help = "Click to go back to file uploads page"):
        st.session_state.page = 'basic_uploads'
        st.rerun()
    if btn2.button("Submit", help = "Click here to submit these images to the model"):  
        if (st.session_state.edited == True): # Modification of uploaded files in case of editing
            uploaded_files_copy = [copy.copy(uploaded_file) for uploaded_file in st.session_state.uploaded_files]
            for i in range(len(uploaded_files_copy)):
                if st.session_state.edited_images[uploaded_files_copy[i].name] != 0:
                    st.session_state.uploaded_files[i] = st.session_state.edited_images[uploaded_files_copy[i].name]
                else:
                    image_data = uploaded_files_copy[i].read()
                    st.session_state.uploaded_files[i] = Image.open(io.BytesIO(image_data))
        st.session_state.page = 'submitted_uploads'
        st.rerun()

# Function to Perform edits on Images
def image_enhance(images_uploaded):
    img_map={img.name:copy.copy(img) for img in images_uploaded}

    if images_uploaded:
        if 'edited_images' not in st.session_state:
            st.session_state.edited_images = {name: 0 for name in img_map.keys()}
        img_selected=st.selectbox("Chcekbox is here ", options=img_map.keys())        
        image = Image.open(io.BytesIO(img_map[img_selected].read()))
        rotation_angles = {name: 0 for name in img_map.keys()}

        # Cropping
        cropped_image = st_cropper(image)
        angle = st.slider('Select rotation angle:', -180, 180, rotation_angles[img_selected])

        # Rotation
        rotation_angles[img_selected] = angle
        rotated_image = cropped_image.rotate(angle, expand=False)
        st.image(rotated_image, caption="Rotated Image", use_column_width=True)

        # Button to apply edits and store the edited image
        if st.button('Apply Edits'):
            st.session_state.edited_images[img_selected] = rotated_image
            st.success("Edits applied and image stored.")
            st.session_state.edited = True
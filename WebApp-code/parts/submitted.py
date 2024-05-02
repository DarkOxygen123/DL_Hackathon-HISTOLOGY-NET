import streamlit as st
from streamlit_image_comparison import image_comparison
from PIL import Image
import io
import copy
from parts.mask_model import mask_images
import zipfile

# Function to execute the "Submit images for segmentation" Action
def submitted_uploads_page():
    if "segmented_files" not in st.session_state:
        st.session_state.segmented_files = None
        st.rerun()

    # Get segmentation masks
    model_path = "model/model_try2_14.pth"
    device = "cpu"
    uploaded_files_copy = [copy.copy(uploaded_file) for uploaded_file in st.session_state.uploaded_files]
    st.session_state.segmented_files = mask_images(uploaded_files_copy, model_path, device)
    del uploaded_files_copy

    # Download and Navigation Buttons
    btn1, btn2, _, btn3 = st.columns(4)
    if btn1.button("Upload New Files"):
        st.session_state.page = "basic_uploads"
        st.session_state.uploaded_files = None
        st.rerun()
    if btn2.button("Visualize Better"):
        st.session_state.page = "compare_uploads"
        st.rerun()
    with btn3:
        uploaded_files_copy = [copy.copy(uploaded_file) for uploaded_file in st.session_state.uploaded_files]
        segmented_files_copy = [copy.copy(segmented_file) for segmented_file in st.session_state.segmented_files]
        download_images(uploaded_files_copy, segmented_files_copy, st.session_state.names)
        del uploaded_files_copy
        del segmented_files_copy

    st.subheader("Segmentation Results: ")
    st.write("")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("<p style='text-align: center;'>Original Image</p>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown("<p style='text-align: center;'>Segmentation</p>", unsafe_allow_html=True)

    # Display Results
    uploaded_files_copy = [copy.copy(uploaded_file) for uploaded_file in st.session_state.uploaded_files]
    segmented_files_copy = [copy.copy(segmented_file) for segmented_file in st.session_state.segmented_files]
    for i in range(len(st.session_state.uploaded_files)):
        try:
            with cols[0]:
                if st.session_state.edited == False:
                    image_data = uploaded_files_copy[i].read()
                    input_image = Image.open(io.BytesIO(image_data))
                else:
                    input_image = uploaded_files_copy[i]
                st.image(input_image, use_column_width=True)
        except Exception as e:
            st.error(f"Error processing {st.session_state.uploaded_files[i].name}: {e}")
        try:
            with cols[1]:
                mask = segmented_files_copy[i]
                st.image(mask, use_column_width=True)
        except Exception as e:
            st.error(f"Error processing Segmentation Mask: {e}")
    del uploaded_files_copy
    del segmented_files_copy

# Function to allow downloading images and masks
def download_images(images, masks, names):
    if st.session_state.edited == False:
        for i in range(len(images)):
            image_data = images[i].read()
            images[i] = Image.open(io.BytesIO(image_data))
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for i, image in enumerate(images):
            file_name=names[i].split('.')[0]
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")
            image_bytes = image_bytes.getvalue()
            zip_file.writestr(f"images/{file_name}.png", image_bytes)
        for i, mask in enumerate(masks):
            file_name=names[i].split('.')[0]
            mask_bytes = io.BytesIO()
            mask.save(mask_bytes, format="PNG")
            mask_bytes = mask_bytes.getvalue()
            zip_file.writestr(f"masks/{file_name}_segmented.png", mask_bytes)    
    zip_buffer.seek(0)
    
    # Create a download link for the zip file
    st.download_button(
        label="Download Results",
        data=zip_buffer,
        file_name="images_and_masks.zip",
        mime="application/zip",
        help="Click here to download a zip file with original images and masks"
    )
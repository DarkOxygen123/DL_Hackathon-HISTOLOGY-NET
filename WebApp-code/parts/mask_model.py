import ternausnet
import ternausnet.models
import streamlit as st
import numpy as np
from PIL import Image
import io
import copy

import torch
from torchvision import transforms

# Function for loading model
def load_model(path, device='cpu'): 
    model = ternausnet.models.UNet11() 
    checkpoint = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    return model

# Function to get segmentation mask for each Image
def get_masked_image(uploaded_file, model, device='cpu'):
    # Converting uploaded_file to PIL object if needed
    if st.session_state.edited == False:
        image_data = uploaded_file.read()
        input_image = Image.open(io.BytesIO(image_data))
    else:
        input_image = uploaded_file
    if input_image.mode != 'RGB':
        input_image = Image.merge("RGB", (input_image, input_image, input_image))
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()  # resizing may be needed
    ])
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # Passing the uploaded_file to the model
    model.eval()  
    with torch.no_grad():
        output_tensor = model(input_tensor)
        output_tensor = torch.sigmoid(output_tensor)
    output_numpy = output_tensor.squeeze().cpu().numpy() # Check for GPU or CPU
    output_numpy = (output_numpy > 0.5).astype(np.uint8)
    output_image = Image.fromarray(output_numpy * 255)

    return output_image

# Function to generate segmentation masks for multiple images
def mask_images(uploaded_files, checkpoint_path, device):
    segmented_files = []
    model = load_model(checkpoint_path, device)
    uploaded_files_copy = [copy.copy(uploaded_file) for uploaded_file in st.session_state.uploaded_files]

    for i,uploaded_file in enumerate(uploaded_files):
        segmented_file = get_masked_image(uploaded_file, model, device)
        if st.session_state.edited == True:
            shape = uploaded_files_copy[i].size
        else:
            image_data = uploaded_files_copy[i].read()
            image = Image.open(io.BytesIO(image_data))
            shape = image.size
        segmented_file = segmented_file.resize(shape) # Ensuring Output is of same size as input
        segmented_files.append(segmented_file)
    del uploaded_files_copy
    return segmented_files
import streamlit as st
from parts.instructions import instructions_page
from parts.side_bar import sb
from parts.uploads_page import basic_uploads_page
from parts.enhance import enhance_uploads_page
from parts.submitted import submitted_uploads_page
from parts.compare import compare_images

# Page Config
st.set_page_config(page_title="HistologyNet", page_icon="📋", initial_sidebar_state="expanded")

# Session-State Variables
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
    st.rerun()
if 'edited' not in st.session_state:
    st.session_state.edited = False
    st.rerun()
if 'names' not in st.session_state:
    st.session_state.names = []
    st.rerun()

# Side Bar
sb()

# Main Content
st.title("HistologyNet")
st.markdown('''##### <span style="color:gray">Self-Supervised Segmentation Masking for Histology Images</span>
            ''', unsafe_allow_html=True)

# Tabs on Home Page
tab_upload, tab_instr = st.tabs(["Upload", "Instructions"])

with tab_instr:
    instructions_page()

with tab_upload:
    if 'page' not in st.session_state:
        st.session_state.page = "basic_uploads"
        st.rerun()
    if st.session_state.page == "basic_uploads":
        basic_uploads_page()
    elif st.session_state.page == "enhance_uploads":
        enhance_uploads_page()
    elif st.session_state.page == "submitted_uploads":
        submitted_uploads_page()
    elif st.session_state.page == "compare_uploads":
        compare_images()

    

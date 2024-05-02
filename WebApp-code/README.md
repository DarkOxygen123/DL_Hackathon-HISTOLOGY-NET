# HistologyNet-WebApp

**HistologyNet** is a self-supervised learning model for binary segmentation of histology images. This is the code for the web-application interface to our model, which allows user to input histology images, edit (if needed, crop/rotate) them and visualize/download their segmented masks.

The application is hosted with Streamlit, and can be accesssed with [this link](https://histonet.streamlit.app/).

## Running the application locally

Install the Python libraries mentioned in `requirements.txt`. In addition, the application requires `streamlit`, `io`, `PIL`, `copy`and `numpy` libraries. 

Paste the model checkpoint (.pth file) inside the model folder. The checkpoint file can be obtained by training the segmentation model, as per the instructions in our project's README. 
(Due to limitation on size of submission allowed, it was not feasible to have a pre-existing .pth file saved.)

To get the application running, add the directory where streamlit is installed to your system's path variabes and run the following command inside the project directory:

```
streamlit run streamlit_app.py
```






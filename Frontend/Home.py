import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from pickle import load
from keras.models import load_model
import utils

xception_model = utils.cnn_model()

# Set up the Streamlit app
st.title("Image Captioning")

# Add a file upload widget and get the uploaded file
uploaded_file = st.file_uploader(
    "Choose an image file", type=["jpg", "jpeg", "png"])

# If a file was uploaded
if uploaded_file is not None:
    # Load the image from the uploaded file
    image = Image.open(uploaded_file)

    tokenizer = load(open("./models/tokenizer.p", "rb"))
    model = load_model('./models/model_7.h5')
    max_length = 32

    photo = utils.extract_features(image, xception_model)
    description = utils.predict_caption(model, tokenizer, photo, max_length)

    st.subheader("Text extracted from the image: {}".format(description))
    st.image(image, caption='Uploaded Image', use_column_width=True)

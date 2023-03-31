import streamlit as st
from PIL import Image

# Set up the Streamlit app
st.title("Image Captioning")

# Add a file upload widget and get the uploaded file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# If a file was uploaded
if uploaded_file is not None:
    # Load the image from the uploaded file
    image = Image.open(uploaded_file)
    
    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    
    # Display the text in a text box
    st.write("Text extracted from the image:")
    # st.text('\n'.join([res[1] for res in result]))



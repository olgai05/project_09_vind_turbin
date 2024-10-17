import streamlit as st
import torch
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Ensure correct path to 'best.pt'

# Title of the app
st.title("YOLO Object Detection")

# Option to upload an image or enter a URL
option = st.selectbox('Select image input method:', ('Upload an image', 'Enter image URL'))

if option == 'Upload an image':
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        img = np.array(image)  # Convert image to numpy array

        # Run inference
        results = model(img)

        # Display results
        st.write("Results:")
        st.dataframe(results.pandas().xyxy[0])  # Display the results in a dataframe

        # Display predictions on image
        st.image(results.render()[0], caption='Predictions', use_column_width=True)

elif option == 'Enter image URL':
    image_url = st.text_input("Enter the URL of the image:")

    if image_url:
        try:
            # Fetch image from URL
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='Downloaded Image', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            # Preprocess the image
            img = np.array(image)  # Convert image to numpy array

            # Run inference
            results = model(img)

            # Display results
            st.write("Results:")
            st.dataframe(results.pandas().xyxy[0])  # Display the results in a dataframe

            # Display predictions on image
            st.image(results.render()[0], caption='Predictions', use_column_width=True)

        except Exception as e:
            st.error(f"Error fetching the image: {e}")

# Run the app using the command below in your terminal:
# streamlit run main.py


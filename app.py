import streamlit as st
import numpy as np
from tensorflow import load_model
from PIL import Image
from tensorflow.keras.losses import MeanSquaredError


# Load the two separate models
gender_model = load_model('gender_model.h5')
age_model = load_model('age_model.h5', custom_objects={'mse': MeanSquaredError()})


# Constants
image_size = 64

def predict_image(image):
    # Resize and normalize the image
    img = image.resize((image_size, image_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions with separate models
    gender_pred = gender_model.predict(img_array)
    age_pred = age_model.predict(img_array)

    # Interpret predictions
    gender = "Female" if np.argmax(gender_pred[0]) == 1 else "Male"
    age = int(age_pred[0][0])

    return gender, age

# Streamlit UI
st.title("👨‍🦱👩‍🦰 Age and Gender Prediction App")
st.write("Upload a face photo and get predicted age and gender instantly!")

uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        gender, age = predict_image(image)
        st.success(f"Predicted Gender: **{gender}**")

        st.success(f"Predicted Age: **{age} years**")


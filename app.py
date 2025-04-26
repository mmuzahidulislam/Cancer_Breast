import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load the model
model = tf.keras.models.load_model("breast_cancer.h5")

# Define class names in the same order as my model's output
class_names = ["Non Cancerous", "Cancerous"]

st.title("Breast Cancer Detection")
st.write("Upload an ultrasound image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and decode the image file as grayscale
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("Image could not be loaded.")
    else:
        # Resize model expects image
        img_resized = cv2.resize(img, (150, 150))


        img_input = np.expand_dims(img_resized, axis=-1)
        img_input = np.expand_dims(img_input, axis=0)

        # Show the uploaded image
        st.image(img_resized, caption="Uploaded Image", use_column_width=True, channels="GRAY")

        # Make a prediction
        prediction = model.predict(img_input)

        # Predicted class index and name
        predicted_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_index]

        # Display results
        st.subheader("Prediction:")
        st.write(f"Prediction Probabilities: {prediction[0]}")
        st.subheader("Predicted Class Index:")
        st.write(predicted_index)
        st.subheader("Predicted Result:")
        st.write(predicted_class)

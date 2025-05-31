import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load your trained model
model = YOLO("best2.pt")

st.title("License Plate Detection")


# File upload
uploaded_file = st.file_uploader("Upload an image of car/cars and I will find your license plate!!", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Run prediction
    results = model.predict(source=image_np, conf=0.25)

    # Get the annotated image
    annotated_img = results[0].plot()

    # Display results
    st.image(annotated_img, caption="Detected License Plate", use_column_width=True)

st.subheader("Model Summary")

st.markdown("Model transfer learned off of YOLOv8/fine tuned for darkness in object detection and trained on 7K images from RoboFlow. ")
st.markdown("**First Train**: Transfer learned off of YOLOv8 model with 7k training images and 2k for validation. ")
st.markdown("**Fine Tuning**: Used openCV for augmentation to account for low-light conditions and trained on a subset of 2k images from the original training set. ")
st.markdown("""
**Evaluation Metrics**
| Version      | Box Precision | Recall | mAP@0.5 | Notes |
|--------------|---------------|--------|---------|-------|
| First Train  | 0.985         | 0.942  | 0.971   | Trained on 7K images for 25 epochs|
| Fine-tuned   | 0.981         | 0.933  | 0.970   | Improved low-light detection |
""")

st.subheader("Improvements")
st.markdown("1. **Cloud**: The model took about 55 hours in total training. Next time, I would use a cloud service like AWS EC2 or SageMaker to speed up training.")
st.markdown("2. **Data**: I would have liked to used more data on low light conditions outside of my augmnetation of 2000 images from the orginal training set, sourced more data for obstructed plates and created a full sized unseen testing dataset.")
st.markdown("3. **Next Step**: With given time I would add a plate reading feature to this project. After detecting a licesne plate, tell me what the plate number is...")

st.markdown("###### For Avolta by Arik")

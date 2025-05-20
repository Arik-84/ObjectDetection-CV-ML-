import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load your trained model
model = YOLO("best2.pt")

st.title("License Plate Detection")
st.markdown("#### *I Love Avolta!!*")
 

# File uploader
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

st.markdown("""
**Description**
| Version      | Box Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Notes |
|--------------|---------------|--------|---------|--------------|-------|
| First Train  | 0.985         | 0.942  | 0.971   | 0.707        | Trained on 7K images for 25 epochs|
| Fine-tuned   | *TBD*         | *TBD*  | *TBD*   | *TBD*        | Will improve low-light detection |
""")

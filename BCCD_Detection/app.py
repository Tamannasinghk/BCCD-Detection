import gradio as gr
from ultralytics import YOLO
import os
import glob
# Load the trained YOLOv10 model
model = YOLO('/content/runs/detect/train/weights/best.pt')

# Inference function for Gradio
def detect_objects(image):
    # Run prediction
    results = model.predict(source=image, save=True, conf=0.5)

    # Get the path of the saved image
    predicted_image_dir = 'runs/detect/predict'
    predicted_image_path = glob.glob(f"{predicted_image_dir}/*")

    return predicted_image_path[0]

# Create the Gradio interface
app = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=gr.Image(type="filepath", label="Detected Image"),
    title="YOLOv10 Object Detection App",
    description="Upload an image to detect blood cells (RBC, WBC, Platelets) using the fine-tuned YOLOv10 model."
)

# Launch the app
app.launch(share=True)

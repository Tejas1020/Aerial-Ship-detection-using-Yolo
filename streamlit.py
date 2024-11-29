import streamlit as st
import torch
from PIL import Image
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil
import cv2
import tempfile
import os

# Set page config
st.set_page_config(page_title="AI Powered Ship Detection using SAR", layout="wide")

def load_model():
    """Load the YOLO model"""
    model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='yolov10x.pt',
        confidence_threshold=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return model

def process_image(image_path, model, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio):
    """Process image using SAHI and YOLO"""
    # Read image
    image = read_image_as_pil(image_path)
    
    # Get predictions
    result = get_sliced_prediction(
        image,
        model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio
    )
    
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    # Draw predictions
    for prediction in result.object_prediction_list:
        bbox = prediction.bbox
        category_name = prediction.category.name
        score = prediction.score.value
        
        # Draw rectangle
        cv2.rectangle(
            image_np,
            (int(bbox.minx), int(bbox.miny)),
            (int(bbox.maxx), int(bbox.maxy)),
            (0, 255, 0),
            2
        )
        
        # Draw label
        label = f"{category_name}: {score:.2f}"
        cv2.putText(
            image_np,
            label,
            (int(bbox.minx), int(bbox.miny) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    return image_np, result.object_prediction_list

def main():
    st.title("AI Powered Ship Detection using SAR")
    
    # Sidebar configuration
    st.sidebar.header("SAHI Configuration")
    
    # Slicing parameters
    st.sidebar.subheader("Slicing Parameters")
    slice_height = st.sidebar.slider(
        "Slice Height",
        min_value=100,
        max_value=1024,
        value=512,
        step=64,
        help="Height of each slice in pixels"
    )
    
    slice_width = st.sidebar.slider(
        "Slice Width",
        min_value=100,
        max_value=1024,
        value=512,
        step=64,
        help="Width of each slice in pixels"
    )
    
    # Overlap parameters
    st.sidebar.subheader("Overlap Parameters")
    overlap_height_ratio = st.sidebar.slider(
        "Height Overlap Ratio",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Overlap ratio between consecutive slices in height"
    )
    
    overlap_width_ratio = st.sidebar.slider(
        "Width Overlap Ratio",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Overlap ratio between consecutive slices in width"
    )
    
    # Add information about parameters
    st.sidebar.markdown("""
    ### Parameter Guide
    - **Slice Size**: Larger slices process more context at once but use more memory
    - **Overlap Ratio**: Higher overlap may catch objects at slice boundaries but increases processing time
    """)
    
    # Load model
    @st.cache_resource
    def get_model():
        return load_model()
    
    model = get_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Create columns for before/after comparison
        col1, col2 = st.columns(2)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Display original image
        original_image = Image.open(uploaded_file)
        col1.header("Original Image")
        col1.image(original_image, use_column_width=True)
        
        # Process image and display results
        with st.spinner('Processing image...'):
            processed_image, predictions = process_image(
                tmp_path,
                model,
                slice_height,
                slice_width,
                overlap_height_ratio,
                overlap_width_ratio
            )
            
            col2.header("Detected Objects")
            col2.image(processed_image, use_column_width=True)
            
            # Display detection results
            st.header("Detection Results")
            
            # Create a container for detection results
            with st.container():
                # Display total count
                total_ships = len(predictions)
                st.markdown(f"**Total Objects Detected:** {total_ships}")
                
                # Display individual detections
                for pred in predictions:
                    st.write(f"- Found {pred.category.name} with confidence {pred.score.value:.2f}")
        
        # Clean up temporary file
        os.unlink(tmp_path)

if __name__ == "__main__":
    main()
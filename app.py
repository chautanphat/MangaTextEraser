import os
import logging
import sys
from flask import Flask, render_template, request, jsonify, current_app
import torch
from PIL import Image
import cv2
import numpy as np
import io
import base64
from manga_utils import prepare_image, create_inpainting_mask

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Model loading function
def load_model():
    try:
        model_path = os.path.join(MODELS_DIR, 'custom_model.pt')
        logger.info(f"Looking for custom model at path: {model_path}")

        if not os.path.exists(model_path):
            error_msg = f"Custoom model file not found at {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Found custom model file, attempting to load...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

        if model is None:
            raise ValueError("Model loading returned None")

        model.eval()
        logger.info("Successfully loaded custom YOLOv5 model")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLOv5 model: {str(e)}")
        raise

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def process_image(image):
    try:
        if model is None:
            raise ValueError("Model is not loaded. Please ensure custom_model.pt is present in the models directory.")

        # Prepare image for processing
        img_array = prepare_image(image)
        logger.debug(f"Processing image of shape {img_array.shape} and type {img_array.dtype}")

        # Run inference
        results = model(img_array)
        logger.debug("Inference completed successfully")

        # Get bounding boxes
        boxes = results.xyxy[0].cpu().numpy()
        logger.debug(f"Detected {len(boxes)} text regions")

        # Create mask for inpainting
        mask = create_inpainting_mask(img_array.shape, boxes)

        # Inpaint the text regions
        # Fill text regions with white
        processed_img = img_array.copy()
        processed_img[mask > 0] = [255, 255, 255]

        return Image.fromarray(processed_img)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if not model:
        return jsonify({'error': 'Model is not loaded. Please ensure custom_model.pt is present in the models directory.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_image_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400

    try:
        # Read and process image
        image = Image.open(file.stream)
        processed_image = process_image(image)

        # Convert processed image to base64
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'success': True,
            'processed_image': f'data:image/png;base64,{img_str}'
        })

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Try to load the model
try:
    model = load_model()
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None  # Set model to None instead of exiting

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
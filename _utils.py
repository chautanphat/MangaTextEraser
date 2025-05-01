import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def prepare_image(image):
    """
    Prepare image for YOLOv5 processing
    """
    try:
        if isinstance(image, Image.Image):
            # Convert PIL Image to numpy array
            image = np.array(image)

        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            logger.debug("Converting grayscale image to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            logger.debug("Converting RGBA image to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:  # Already RGB
            logger.debug("Image is already in RGB format")
        else:
            raise ValueError(
                f"Unexpected image format with shape {image.shape}")

        return image
    except Exception as e:
        logger.error(f"Error in prepare_image: {str(e)}")
        raise


def create_inpainting_mask(image_shape, boxes):
    """
    Create a mask for inpainting based on detected text boxes
    """
    try:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        logger.debug(
            f"Creating mask with shape {mask.shape} for {len(boxes)} boxes")

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image_shape[1], x2)
            y2 = min(image_shape[0], y2)

            if x1 < x2 and y1 < y2:  # Valid box
                pad = 2  # Increased padding for better coverage
                x1_pad = max(0, x1 - pad)
                y1_pad = max(0, y1 - pad)
                x2_pad = min(image_shape[1], x2 + pad)
                y2_pad = min(image_shape[0], y2 + pad)
                cv2.rectangle(mask, (x1_pad, y1_pad), (x2_pad, y2_pad), 255,
                              -1)
            else:
                logger.warning(
                    f"Skipping invalid box coordinates: {(x1, y1, x2, y2)}")

        return mask
    except Exception as e:
        logger.error(f"Error in create_inpainting_mask: {str(e)}")
        raise

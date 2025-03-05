import cv2
import numpy as np
from .otsu import otsu


def watershed(image):
    """
    Apply Watershed segmentation using Otsu's thresholding for initial foreground/background separation.

    Args:
        image: numpy.ndarray
            The grayscale image to be segmented.

    Returns:
        binary_segmentation: numpy.ndarray
            The binary segmentation image (foreground=0, background=255).
    """
    thresholded_image, _ = otsu(image)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresholded_image, kernel, iterations=3) 
    sure_fg = cv2.erode(thresholded_image, kernel, iterations=3)  
    unknown = cv2.subtract(sure_bg, sure_fg)  

    # define markers for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1  
    markers[unknown == 255] = 0  

    # convert to color for watershed
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Apply watershed
    cv2.watershed(image_color, markers)

    # Boundaries
    # binary_segmentation = np.where(markers == -1, 255, 0).astype(np.uint8)

    # Segmentation
    binary_segmentation = np.ones_like(image) * 255
    binary_segmentation[markers == 1] = 0
    binary_segmentation[markers == -1] = 255
    
    return binary_segmentation
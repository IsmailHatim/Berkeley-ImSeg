import cv2
import numpy as np
from .otsu import otsu

def post_process_watershed(image, markers):
    """
    Improve watershed segmentation by refining the mask.

    Args:
        image: numpy.ndarray
            Original image in color.
        markers: numpy.ndarray
            Watershed output markers.

    Returns:
        binary_mask: numpy.ndarray
            Final binary mask for segmentation.
    """    
    mask = np.zeros_like(markers, dtype=np.uint8)
    mask[markers > 1] = 255 
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h, w = mask.shape[:2]
    mask_floodfill = mask.copy()
    flood_mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(mask_floodfill, flood_mask, (0, 0), 255)
    mask_filled = cv2.bitwise_or(mask, cv2.bitwise_not(mask_floodfill))

    return mask_filled


def watershed(image):
    """
    Apply Watershed algorithm to segment objects in an image.

    Args:
        image: str
            Iinput image in grayscale.

    Returns:
        segmented_image: numpy.ndarray
            The segmented image with contours drawn.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Smoothing with open morph
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Define sure Background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Define sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.15 * dist_transform.max(), 255, 0)  # Avant : 0.2, maintenant 0.15

    # *Define unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Define markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers.astype(np.int32)
    markers += 1
    markers[unknown == 255] = 0

 
    cv2.watershed(image, markers)
    segmented_binary = post_process_watershed(image, markers)

    return segmented_binary
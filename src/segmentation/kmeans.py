import numpy as np
import cv2
from sklearn.cluster import KMeans

def kmeans(image, k=2):
    """
    Perform image segmentation using K-means clustering

    Args:
        image: numpy.ndarray
            The grayscale image to be tresholded
        
        k: int
            The number of clusters (by default 2 because we want a binary segmentation)
        
    Returns:
        tresholded_image: numpy.ndarray
            The binary image resulting from the tresholdind process, where
            foreground pixels are set to 255 and background pixels to 0
    """
    if len(image.shape) != 2:
        raise ValueError('Input image must be a grayscale image (2D array)')
        
    pixels = image.reshape((-1, 1))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)

    segmented_image = labels.reshape(image.shape)
    segmented_image = (segmented_image * 255 // (k-1)).astype(np.uint8)

    return cv2.bitwise_not(segmented_image)


    
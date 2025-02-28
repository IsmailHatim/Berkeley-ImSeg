import cv2
import numpy as np

def otsu_treshold(image):

    """
    Apply Otsu's tresholding method to segment the image into foreground and background.

    Args:
        image: numpy.ndarray
            The grayscale image to be tresholded

    Returns:
        tresholded_image: numpy.ndarray
            The binary image resulting from the tresholding process, where foreground pixels are set to 255
            and background pixels are set to 0.

        best_treshold: int
            The optimal treshold value determined by Otsu's method 
    """

    if len(image.shape) != 2:
        raise ValueError('Input image must be a grayscale image (2D array)')
    
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256)) # image.ravel() flattens the whole image

    total_pixels = image.size
    prob = hist / total_pixels

    mean_global = np.sum(np.arange(256) * prob)

    best_treshold = 0
    max_variance = 0
    sum_class1 = 0
    prob_class1 = 0

    for t in range(256):
        
        prob_class1 += prob[t]
        if prob_class1 == 0 or prob_class1==1:
            continue
            
        sum_class1 += t*prob[t]
        mean_class1 = sum_class1 / prob_class1
        mean_class2 = (mean_global - sum_class1) / (1 - prob_class1)

        variance_between = prob_class1 * (1-prob_class1) * (mean_class1 - mean_class2) ** 2

        if variance_between > max_variance:
            max_variance = variance_between
            best_treshold = t
    
    _, tresholded_image = cv2.threshold(image, best_treshold, 255, cv2.THRESH_BINARY)
    
    return tresholded_image, best_treshold
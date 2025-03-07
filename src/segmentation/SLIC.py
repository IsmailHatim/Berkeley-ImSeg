import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops_table
from skimage.color import rgb2lab

def post_process_SLIC(image, slic_labels):
    """
    Convert SLIC superpixels into a binary segmentation mask.

    Args:
        image: numpy.ndarray
            Input RGB image.
        slic_labels: numpy.ndarray
            Superpixel label map from SLIC.

    Returns:
        binary_mask: numpy.ndarray
            Binary segmentation mask where 255 = object, 0 = background.
    """
    image_lab = rgb2lab(image)

    properties = regionprops_table(slic_labels, intensity_image=image_lab[:, :, 0], properties=['label', 'mean_intensity'])
    threshold, _ = cv2.threshold(image_lab[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = np.zeros_like(slic_labels, dtype=np.uint8)

    for label, mean_intensity in zip(properties['label'], properties['mean_intensity']):
        if mean_intensity < threshold:  
            mask[slic_labels == label] = 255

    return mask

def SLIC(image, num_segments=200, compactness=10, max_iter=10):
    """
    Apply SLIC (Simple Linear Iterative Clustering) superpixel segmentation using K-Means.

    Args:
        image: numpy.ndarray
            The input RGB image.
        num_segments: int
            The number of desired superpixels.
        compactness: float
            The balance between color similarity and spatial proximity.
        max_iter: int
            The maximum number of iterations for K-Means.

    Returns:
        segmented_image: numpy.ndarray
            Image with segmented superpixels.
        labels: numpy.ndarray
            Superpixel label map.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 1:
        image = cv2.merge([image, image, image])

    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


    height, width, _ = image.shape
    num_pixels = height * width
    S = int(np.sqrt(num_pixels / num_segments))

    centers = []
    for y in range(S // 2, height, S):
        for x in range(S // 2, width, S):
            centers.append((x, y))

    centers = np.array(centers)
    pixel_coords = np.indices((height, width)).transpose(1, 2, 0).reshape(-1, 2)
    pixel_colors = image_lab.reshape(-1, 3)
    features = np.hstack((pixel_colors, compactness * pixel_coords))
    kmeans = KMeans(n_clusters=len(centers), max_iter=max_iter, n_init=1, random_state=42)
    labels = kmeans.fit_predict(features)
    segmented_image = (mark_boundaries(image, labels.reshape(height, width)) * 255).astype(np.uint8)
    mask = post_process_SLIC(segmented_image, labels.reshape(height, width))

    return segmented_image, mask, labels.reshape(height, width)
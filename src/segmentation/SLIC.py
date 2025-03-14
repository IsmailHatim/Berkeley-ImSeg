import cv2
import numpy as np
from sklearn.cluster import KMeans


def initialize_centers(image, S):
    """ Initialize superpixels centers on a regular grid """
    h, w = image.shape
    centers = []
    for y in range(S // 2, h, S):
        for x in range(S // 2, w, S):
            centers.append([y, x, image[y, x]])
    return np.array(centers)

def compute_distance(pixel, center, m, S):
    """ Compute distance between a pixel and a center taking into acount intensity and space """
    intensity_dist = abs(pixel[2] - center[2])  
    spatial_dist = np.linalg.norm(pixel[:2] - center[:2]) 
    return np.sqrt(intensity_dist*2 + (m / S) ** 2 * spatial_dist*2)

def slic_segmentation(image, num_superpixels=300, m=5, max_iter=10):
    """
    Apply the SLIC (Simple Linear Iterative Clustering) segmentation algorithm.

    Args:
        image: numpy.ndarray
            The grayscale image to be segmented.
        num_superpixels: int, optional
            The number of desired superpixels (default is 100).
        m: int, optional
            The compactness parameter controlling the balance between spatial and intensity distance (default is 10).
        max_iter: int, optional
            The maximum number of iterations for cluster refinement (default is 10).

    Returns:
        labels: numpy.ndarray
            The labeled image where each pixel is assigned a superpixel label.
        centers: numpy.ndarray
            The array containing the updated center positions of the superpixels.
    """

    h, w = image.shape
    S = int(np.sqrt((h * w) / num_superpixels)) 

    centers = initialize_centers(image, S)

    labels = -np.ones((h, w), dtype=np.int32)
    distances = np.full((h, w), np.inf)

    for _ in range(max_iter):
        for i, center in enumerate(centers):
            cy, cx = int(center[0]), int(center[1])

            for dy in range(-S, S):
                for dx in range(-S, S):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        pixel = [ny, nx, image[ny, nx]]
                        d = compute_distance(pixel, center, m, S)

                        if d < distances[ny, nx]:
                            distances[ny, nx] = d
                            labels[ny, nx] = i

        new_centers = np.zeros_like(centers)
        counts = np.zeros(len(centers))

        for y in range(h):
            for x in range(w):
                idx = labels[y, x]
                new_centers[idx] += np.array([y, x, image[y, x]])
                counts[idx] += 1

        centers = new_centers / counts[:, None]
    
    return labels, centers

def SLIC(image):
    """ Separate foreground and background using mean intensity of superpixels """
    labels, centers = slic_segmentation(image)
    num_superpixels = len(centers)

    superpixel_intensity = np.zeros(num_superpixels)
    counts = np.zeros(num_superpixels)

    h, w = image.shape
    for y in range(h):
        for x in range(w):
            idx = labels[y, x]
            superpixel_intensity[idx] += image[y, x]
            counts[idx] += 1

    superpixel_intensity /= counts 

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(superpixel_intensity.reshape(-1, 1))

    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            idx = labels[y, x]
            mask[y, x] = 0 if kmeans.labels_[idx] == 1 else 255

    return mask
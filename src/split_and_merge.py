import cv2
import numpy as np
from .otsu import otsu

def split_and_merge(image, threshold=20):
    """
    Apply the Split and Merge segmentation algorithm.

    Args:
        image: numpy.ndarray
            Grayscale image to be segmented.
        threshold: int
            Maximum allowed variance for a region to be considered homogeneous.

    Returns:
        segmented_image: numpy.ndarray
            The segmented image.
    """
    original_height, original_width = image.shape

    def split(image, x, y, size):
        """ Récursive function to split regions if they are not homogeneous. """
        region = image[y:y+size, x:x+size]
        var = np.var(region)  

        if var < threshold or size < 2:
            return [(x, y, size)]

        half_size = size // 2
        return (
            split(image, x, y, half_size) +
            split(image, x + half_size, y, half_size) +
            split(image, x, y + half_size, half_size) +
            split(image, x + half_size, y + half_size, half_size)
        )

    def merge(image, regions):
        """ Function to merge similar regions. """
        segmented = np.zeros_like(image)

        for x, y, size in regions:
            region = image[y:y+size, x:x+size]
            mean_value = int(np.mean(region))
            segmented[y:y+size, x:x+size] = mean_value

        return segmented

    height, width = image.shape
    max_size = 2 ** int(np.floor(np.log2(min(height, width))))

    image = cv2.resize(image, (max_size, max_size))

    regions = split(image, 0, 0, max_size)

    segmented_image = cv2.resize(merge(image, regions), (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    thresholded_image, best_t = otsu(segmented_image)

    return thresholded_image

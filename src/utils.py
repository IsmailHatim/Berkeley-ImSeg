import numpy as np
import cv2
import scipy
from sklearn.metrics import jaccard_score


def load_image(path, colorspace=cv2.IMREAD_GRAYSCALE):

    """
    Load an image from the specified path and convert it to grayscale.

    Args:
        path: str
            The file path to the image to be loaded
    Returns:
        image: numpy.ndarray
        The grayscaled image loaded from the specific path
    """

    image = cv2.imread(path, colorspace)
    
    if image is None:
        raise FileNotFoundError(f'The file at path {path} does not exist or could not be loaded')
    
    return image


def load_gt(gt_path):

    """
    Load the ground truth segmentation from a .mat file and convert it into a binary segmentation image.

    Args:
        gt_path: str
            The file path to the .mat file containing the ground truth data.
        
    Returns:
        gt_segmentation_bin: numpy.ndarray
            A binary image representing the ground truth segmentation, where foreground pixels are
            set to 255 and background pixels are set to 0.
    """
    
    data = scipy.io.loadmat(gt_path)
    
    if data is None:
        raise FileNotFoundError(f'The file at path {gt_path} does not exist or could not be loaded')

    # Extract ground truth segmentation
    gt = data['groundTruth'][0][0]
    gt = gt['Segmentation'][0][0]

    gt_segmentation_bin = (gt > 1).astype(np.uint8) * 255 

    gt_segmentation_bin = 255 - gt_segmentation_bin 

    return gt_segmentation_bin


def compute_iou_score(image1, image2):

    """
    Compute the IoU similarity coefficient (Intersection on Union) between two binary images.

    Args:
        image1: numpy.ndarray
            The first binary image to be compared
        
        image2: numpy.ndarray
            The second binary image to be compared
    
    Returns:
        iou: float
            The IoU similarity coefficient between the two images.

    """
    
    image1 = (image1 == 255).astype(np.uint8)
    image2 = (image2 == 255).astype(np.uint8)

    iou = round(jaccard_score(image1.flatten(), image2.flatten(), average='binary').item(), 4)

    return iou

def extract_edges(binary_image):
    """
    Extract edges from a binary mask using morphological operations.

    Args:
        binary_image: numpy.ndarray
            The binary image (0 and 255 values).

    Returns:
        edges: numpy.ndarray
            The binary edge image.
    """
    kernel = np.ones((3, 3), np.uint8)
    
    # Apply dilation and erosion
    dilated = cv2.dilate(binary_image, kernel, iterations=1)
    eroded = cv2.erode(binary_image, kernel, iterations=1)
    
    # Compute the boundary (morphological gradient)
    edges = cv2.absdiff(dilated, eroded)

    return edges

def compute_boundary_recall(image1, image2, tolerance=2):
    """
    Compute the Boundary Recall between two binary images.

    Args:
        image1: numpy.ndarray
            The first binary image (ground truth).
        
        image2: numpy.ndarray
            The second binary image (prediction).
        
        tolerance: int, optional
            The distance tolerance to match boundaries, default is 2 pixels.

    Returns:
        recall: float
            The boundary recall score between the two images.
    """
    binary1 = (image1 == 255).astype(np.uint8)
    binary2 = (image2 == 255).astype(np.uint8)
    
    edges1 = extract_edges(binary1)
    edges2 = extract_edges(binary2)
    
    y1, x1 = np.where(edges1 > 0)
    y2, x2 = np.where(edges2 > 0)
    
    if len(x2) == 0:
        return 0.0
    
    # Compute distance transform from predicted edges
    dist_transform = cv2.distanceTransform(edges1, cv2.DIST_L2, 5)
    
    # Count the number of ground truth edges within tolerance
    matched = np.sum(dist_transform[y2, x2] <= tolerance)
    
    recall = round(matched / len(x2), 4)
    
    return recall
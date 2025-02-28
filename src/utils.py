import numpy as np
import cv2
import scipy
from sklearn.metrics import jaccard_score


def load_image(path):

    """
    Load an image from the specified path and convert it to grayscale.

    Args:
        path: str
            The file path to the image to be loaded
    Returns:
        image: numpy.ndarray
        The grayscaled image loaded from the specific path
    """

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
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
    
    # Load the .mat file
    data = scipy.io.loadmat(gt_path)
    
    if data is None:
        raise FileNotFoundError(f'The file at path {gt_path} does not exist or could not be loaded')

    # Extract ground truth segmentation
    gt = data['groundTruth'][0][0]
    gt = gt['Segmentation'][0][0]

    # Convert the segmentation into binary format
    gt_segmentation_bin = (gt > 1).astype(np.uint8) * 255 

    # match the foreground and background with Otsu's output
    gt_segmentation_bin = 255 - gt_segmentation_bin 

    return gt_segmentation_bin


def compute_jaccard_score(image1, image2):

    """
    Compute the Jaccard similarity coefficient (Intersection on Union) between two binary images.

    Args:
        image1: numpy.ndarray
            The first binary image to be compared
        
        image2: numpy.ndarray
            The second binary image to be compared
    
    Returns:
        iou: float
            The Jaccard similarity coefficient between the two images.

    """
    
    # Convert images to binary format (foreground=1 and background=0))
    image1 = (image1 == 255).astype(np.uint8)
    image2 = (image2 == 255).astype(np.uint8)

    # Compute Jaccard score
    iou = round(jaccard_score(image1.flatten(), image2.flatten(), average='binary').item(), 4)

    return iou
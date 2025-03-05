import cv2
import numpy as np

def region_growing(image, seed, threshold=10):
    """
    Apply the Region Growing segmentation algorithm.

    Args:
        image: numpy.ndarray
            The grayscale image to be segmented.
        seed: tuple (x, y)
            The starting pixel (seed) for region growing.
        threshold: int, optional
            The maximum intensity difference allowed to include a pixel in the region (default is 10).

    Returns:
        segmented_image: numpy.ndarray
            The segmented binary image where the detected region is white (255) and the rest is black (0).
    """

    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image (2D array)")

    # Dimensions de l'image
    height, width = image.shape

    # Créer une image binaire de sortie
    segmented_image = np.zeros_like(image, dtype=np.uint8)

    # Obtenir l'intensité du point de départ
    seed_value = image[seed[1], seed[0]]

    # Liste des pixels à explorer (FIFO queue)
    stack = [seed]

    # Définitions des directions de voisinage (8-connectivité)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while stack:
        x, y = stack.pop()

        # Vérifier si le pixel est déjà segmenté
        if segmented_image[y, x] == 255:
            continue

        # Marquer le pixel comme segmenté
        segmented_image[y, x] = 255

        # Explorer les voisins
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Vérifier si le voisin est dans l'image
            if 0 <= nx < width and 0 <= ny < height:
                # Vérifier la condition d'homogénéité
                if segmented_image[ny, nx] == 0 and abs(int(image[ny, nx]) - int(seed_value)) <= threshold:
                    stack.append((nx, ny))

    return segmented_image

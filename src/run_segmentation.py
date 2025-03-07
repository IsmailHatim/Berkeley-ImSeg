import matplotlib.pyplot as plt
import sys
import os
import importlib
import time
from argparse import ArgumentParser
from utils import load_image, load_gt, compute_jaccard_score, compute_boundary_recall

sys.dont_write_bytecode = True

def main(args):
    ROOT = os.getcwd()
    INDEX = args.index
    image_path = f'{args.image_path}\{INDEX}.jpg'
    ground_path = f'{args.ground_path}\{INDEX}.mat'
    method = getattr(importlib.import_module(f'segmentation.{args.method}'), args.method)

    original_image = load_image(image_path)
    
    start_time = time.time()
    segmented_image = method(original_image)
    end_time = time.time()
    execution_time = end_time - start_time

    gt_segmentation_bin = load_gt(ground_path)

    jaccard_score = compute_jaccard_score(segmented_image, gt_segmentation_bin)
    recall_score = compute_boundary_recall(segmented_image, gt_segmentation_bin)
    
    print("+" + "-" * 50 + "+")
    print(f"Jaccard Score : {jaccard_score}")
    print(f"Boundary Recall Score : {recall_score}")
    print(f"Execution Time : {execution_time:.4f} seconds")
    print("+" + "-" * 50 + "+")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1), plt.imshow(original_image, cmap='gray'), plt.title('Original Image')
    plt.subplot(1, 3, 2), plt.imshow(segmented_image, cmap='gray'), plt.title('Segmented Image')
    plt.subplot(1, 3, 3), plt.imshow(gt_segmentation_bin, cmap='gray'), plt.title('Ground Truth Segmentation')
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser(description='Run selected segmentation method on Berkeley Dataset.')

    parser.add_argument('--method', default='otsu', type=str, help='Segmentation method to use.')
    parser.add_argument('--index', default='24063', type=int, help='Image index from Berkeley Dataset.')
    parser.add_argument('--image_path', default='data/images/train', type=str, help='Images path.')
    parser.add_argument('--ground_path', default='data/ground_truth/train', type=str, help='Ground truth path.')

    args = parser.parse_args()
    
    main(args)

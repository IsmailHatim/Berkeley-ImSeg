# Berkeley-ImSeg Pipeline

![Pipeline Overview](path/to/pipeline_image.png)

## Overview
The **Berkeley-ImSeg** pipeline is an image segmentation project that implements and compares six classical segmentation methods. It is designed for evaluating segmentation performance on natural images using the **BSDS500** dataset.

## Implemented Segmentation Methods
- **Otsu's Method**: Automatic thresholding technique based on inter-class variance maximization.
- **K-Means Clustering**: Unsupervised clustering method that partitions an image into K groups.
- **Region Growing**: Expands regions starting from seed points based on similarity criteria.
- **SLIC (Simple Linear Iterative Clustering)**: Generates superpixels for better spatial consistency.
- **Split-and-Merge**: Hierarchical approach that divides and merges regions based on homogeneity.
- **Watershed Algorithm**: Treats an image as a topographic surface and finds segmentation basins.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Berkeley-ImSeg.git
   cd Berkeley-ImSeg
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the segmentation pipeline using the following command:
```bash
python src/run_segmentation.py --method 'otsu' --index '24063' --image_path "path/to/images" --ground_path "path/to/ground_truth"
```

### Parameters:
- `--method`: The segmentation method to use (`otsu`, `kmeans`, `region_growing`, `SLIC`, `split_and_merge`, `watershed`).
- `--index`: Image index from the dataset.
- `--image_path`: Path to the directory containing input images.
- `--ground_path`: Path to the directory containing ground-truth segmentations.

## Dataset
The project uses the **BSDS500 (Berkeley Segmentation Dataset)**, a dataset containing 500 natural images with ground truth segmentation annotations.

## Evaluation Metrics
The pipeline includes evaluation metrics to compare segmentation performance:
- **IoU (Intersection over Union)**: Measures overlap between predicted and ground-truth segments.
- **Boundary Recall (BR)**: Assesses how well detected edges align with the ground truth.
- **Execution Time**: Benchmarks algorithm speed.

## Contributing
If you would like to contribute, feel free to fork the repository, create a new branch, and submit a pull request.
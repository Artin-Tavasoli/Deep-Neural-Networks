

## Object Detection Using Sparse R-CNN

- Implemented a lightweight Sparse R-CNN model from scratch to detect and localize geometric shapes within a custom synthetic dataset.

![[assets/Model-Overview.png]]

### ⚙️ Methodology

- **Data Processing:** Generated a synthetic dataset of 1000 images (128x128 pixels, black background) containing random circles, squares, and triangles. Applied normalization and constructed a custom `collate_fn` within the dataloader to handle variable-length bounding box targets.
    
- **Architecture:** Constructed a 3-layer Convolutional Neural Network (CNN) backbone to extract 16x16 feature maps with 128 channels.
    
- **Learnable Proposals:** Eliminated the traditional dense Region Proposal Network (RPN) in favor of 20 learnable proposal boxes and features (`nn.Parameter`) that act as a statistical prior and optimize directly during backpropagation.
    
- **Iterative Refinement:** Built a 3-stage dynamic head utilizing ROI Align to extract 7x7 spatial features. These features interact dynamically with the proposal features via batch matrix multiplication to iteratively refine bounding box coordinates and class predictions.
    
- **Bipartite Matching & Loss:** Utilized the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) for strict one-to-one matching between predictions and ground truths, successfully removing the need for Non-Maximum Suppression (NMS). The combined loss function minimized Focal Loss (to handle class imbalance), L1 Loss, and Generalized IoU (GIoU) Loss (for scale-invariant bounding box regression).
    

### Results

The model was trained using the Adam optimizer with a Cosine Annealing learning rate scheduler for 50 epochs. The network demonstrated rapid learning of shape patterns and background differentiation within the first 10 epochs.

- **Quantitative Evaluation (mAP):** Evaluated on a holdout set of 100 test images, the model achieved a Mean Average Precision (mAP @ IoU=0.50:0.95) of **`0.8476`**.
    
- **Qualitative Performance:** The network successfully drew tight bounding boxes around objects, accurately classified shapes, and effectively resolved instances of overlapping geometric figures without producing duplicate boxes.
    

- **Proposal Evolution:** Visual analysis confirmed that the 20 learnable boxes transitioned from completely random and chaotic initializations to uniformly distributed "learned" priors, and finally to precise object localizations after passing through the dynamic head.


![[assets/detection.PNG]]
![[assets/proposal-evolution.PNG]]
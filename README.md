# Enhanced-Convolutional-Neural-Network
High-dimensional and noisy datasets pose challenges for reliable classification, particularly in applications such as neuroimaging and facial recognition. In this work, we investigate an integrated machine learning pipeline that incorporates preprocessing steps (including denoising and data augmentation) and combines convolutional neural networks (CNNs) with encoder–decoder architectures and support vector machines (SVMs). To address class imbalance and data heterogeneity, we employ a multi-teacher knowledge distillation (KD) strategy, where multiple teacher models, each trained on different subsets or aspects of the data, provide guidance to a student model. This setup aims to transfer complementary information and improve performance on underrepresented classes. Experiments on neuroimaging and facial recognition datasets suggest that this combined approach enhances classification consistency and achieve competitive accuracy compared to baseline methods.

Creating machine learning models is a very delicate process that at times can be very data specific. The pipeline was anaylized by creating various models to evaluate the addition of each elements, including preprocessing (denoisng, normalizing and augmentation), Encoder/Decoder architecture, postprocessing (SVM classifier) and Knowledge distillation. Resulting in four different models for each dataset to compare. 
| Item | Description |
| :--- | :--- |
| 1 | Created from unprocessed data utilizing a regular CNN |
| 2 | Created from processed data utilizing a regular CNN |
| 3 | Created from processed data utilizing a CNN with encoder/decoder architecture |
| 4 | Created from processed data utilizing Model 3 as a teacher via knowledge distillation |

The data utilized in this research are ABIDE I (2016), Labeled Faces In the Wild (LFW)(2017) and Olivetti Facial Recognition (1994). 
| Dataset | Example Observation | Study Focus | Subject Details |
| :--- | :--- | :--- | :--- |
| **ABIDE I** | <img width="205" alt="Abide Color 1" src="https://github.com/user-attachments/assets/c2d40723-ff35-4ec7-ab29-e34784cb0cea" align="top" /> <img width="44" alt="ABIDE color scale" src="https://github.com/user-attachments/assets/31dcfcf6-9574-48a5-be3c-2c1877a76593" align="top" /> | f-MRI for ASD | 857 observations |
| **LFW** | <img width="174" alt="LFW Single face" src="https://github.com/user-attachments/assets/cadea331-ef6d-4e06-bf27-38c31f0cdc82" /> | Facial recognition | 5,479 individuals |
| **Olivetti** | <img width="173" alt="Olivetti single face" src="https://github.com/user-attachments/assets/ed434075-4c60-4006-985e-b5f874ee6b91" /> | Facial recognition | 40 individuals |


Included is the code created to process the data using out pipline the full pipeline is described in the following figure: <img width="2350" height="1824" alt="Model structure with utilized images" src="https://github.com/user-attachments/assets/c0fe2c27-d67a-4a64-aca2-a5a40b87e6b5" />

The repository is organized into functional modules. Each script is designed to handle a specific stage of the machine learning pipeline, from raw data denoising to multi-teacher distillation.

### 1. Data Preprocessing & Preparation
* **`mppca_denoise.py`**: Implements a sliding-window MP-PCA denoiser for 3D image data. (Adapted to Python from the original MATLAB code by NeuroPhysics at CFIN).
* **`norm_abide.py` / `run_norm_abide.py`**: Handles per-patient normalization (Z-score or Min-Max) for the ABIDE connectivity matrices, including symmetric data augmentation.
* **`lfw_subset_creator.py` / `combine_lfw.py`**: Tools for managing the LFW dataset, including splitting by image count and merging augmented `.pt` files.
* **`shift_labels.py`**: Ensures label consistency when combining datasets by shifting label indices to prevent overlap between subsets.

### 2. Model Architectures & Training
* **`cnn_ed.py`**: Trains the CNN with an encoder-decoder architecture. It extracts high-dimensional embeddings and saves them to HDF5 format for downstream classification.
* **`regular_cnn.py`**: A baseline training script for a standard CNN used to compare performance against the encoder-decoder and KD models.
* **`dual_teacher.py`**: The core **Knowledge Distillation** script. It utilizes two pre-trained teachers ($T_1$ and $T_2$) and implements **selective distillation**, where the student learns from the "expert" teacher for a specific sample.

### 3. Evaluation & Analysis
* **`lfw_consistent_plots_svm.py` / `consistent_plots_svm.py`**: Trains an SVM classifier on the CNN-derived embeddings and evaluates performance using Accuracy, F1-score, and UMAP decision space visualizations.
* **`model_eval_KD.py` / `model_eval_cnn_ed.py`**: Comprehensive evaluation suites that compute classification metrics and clustering metrics (Silhouette, Davies-Bouldin, and Calinski-Harabasz).
* **`compare_clustering.py`**: A visualization tool to compare the performance of PCA, t-SNE, and UMAP across different embedding types.
* **`consistent_plots_CNN.py` / `lfw_consistent_plots_cnn.py`**: Generates standardized UMAP visualizations to ensure visual consistency across different experimental runs.

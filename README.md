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
| Dataset | Study Focus | Design & Scope | Subject/Observation Details |
| :--- | :--- | :--- | :--- |
| **ABIDE I** | f-MRI data for Autism Spectrum Disorder (ASD) | Connectivity matrices constructed from a variety of locations | 857 observations: 463 with ASD, 403 Control |
| **LFW** | Facial recognition in the midst of class imbalance | Variety of settings including 5,479 unique people | 5,479 individuals: ranging from 1 image to 530 images per person |
| **Olivetti** | Facial recognition without class imbalance | Variety of settings including 40 unique people | 40 individuals: 10 images per person (400 total images) |

Our Pipeline is described in the following figure: <img width="2350" height="1824" alt="Model structure with utilized images" src="https://github.com/user-attachments/assets/c0fe2c27-d67a-4a64-aca2-a5a40b87e6b5" />

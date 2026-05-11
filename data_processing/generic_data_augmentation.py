import numpy as np
import torch
from torch.utils.data import Dataset

'''This script defines a generic additive guasian noise augmentation method. '''

def augment_images( images, labels, augmentation_factor):
        augmented_images = []
        augmented_labels = []

        for image, label in zip(images, labels):
            # Original image
            augmented_images.append(image)
            augmented_labels.append(label)

            for _ in range(augmentation_factor):
                # Add Gaussian noise
                noise = np.random.normal(0, 0.02, image.shape)  # Reduced noise for normalized data
                augmented_image = image + noise

                # Ensure valid range
                augmented_image = np.clip(augmented_image, -3, 3)  # Wider range for normalized data

                augmented_images.append(augmented_image)
                augmented_labels.append(label)

        return np.array(augmented_images), np.array(augmented_labels)


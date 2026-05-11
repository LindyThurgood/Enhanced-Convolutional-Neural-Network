import torch
import torchvision.transforms as T

'''This code defines the image augmentation pattern for the training group augmentation. '''
def perform_torchvision_augmentation(images, labels, factor):
    # Setup the pipeline
    augmenter = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    aug_images = []
    aug_labels = []

    print(f"Generating {len(images) * factor} torchvision-augmented images...")
    
    for i in range(len(images)):
        img = images[i]
        label = labels[i]
        
        # Add the original image once if you want it included in the total
        aug_images.append(img)
        aug_labels.append(label)
        
        # Add the augmented variations
        for _ in range(factor - 1):
            aug_images.append(augmenter(img))
            aug_labels.append(label)

    return torch.stack(aug_images), torch.stack(aug_labels)

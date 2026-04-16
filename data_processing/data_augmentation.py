import numpy as np
import h5py

"""This script applies multiple augmentation techniques to increase variety. Returns: Original + Flip + Noise + Brightness + Shift (5x Data)"""
#Augmentation Function
def augment_dataset(images, labels):
    
    augmented_images = []
    augmented_labels = []
    
    print("Starting augmentation suite...")
    
    for img, label in zip(images, labels):
        # --- 1. Original ---
        augmented_images.append(img)
        augmented_labels.append(label)
        
        # --- 2. Horizontal Flip ---
        # Faces are generally symmetric; this is a high-value augmentation
        flipped_img = np.flip(img, axis=1)
        augmented_images.append(flipped_img)
        augmented_labels.append(label)
        
        # --- 3. Gaussian Noise ---
        # Adds robustness against sensor grain/low light
        noise = np.random.normal(0, 0.03, img.shape)
        noisy_img = np.clip(img + noise, -3, 3)
        augmented_images.append(noisy_img)
        augmented_labels.append(label)
        
        # --- 4. Random Brightness ---
        # Simulates different lighting conditions
        brightness_factor = np.random.uniform(-0.4, 0.4)
        bright_img = np.clip(img + brightness_factor, -3, 3)
        augmented_images.append(bright_img)
        augmented_labels.append(label)
        
        # --- 5. Small Translation (Shift) ---
        # Prevents the model from over-relying on exact pixel locations
        shift_h, shift_w = np.random.randint(-3, 4, size=2)
        shifted_img = np.roll(img, shift_h, axis=0)
        shifted_img = np.roll(shifted_img, shift_w, axis=1)
        augmented_images.append(shifted_img)
        augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)


def main():
    input_name = 'lfw_subset_9.h5' 
    normalize_method = 'zscore'
    
    print(f"Loading {input_name}...")
    try:
        with h5py.File(input_name, 'r') as f_in:
            images = f_in['images'][:] 
            labels = f_in['labels'][:].squeeze()
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Original dataset shape: {images.shape}")

    aug_images, aug_labels = augment_dataset(images, labels)
    print(f"Augmented dataset shape: {aug_images.shape}")

    output_name = 'lfw_9_test_augmented.h5'
    print(f"Saving to {output_name}...")
    
    try:
        with h5py.File(output_name, 'w') as f_out:
            f_out.create_dataset('images', data=aug_images, compression='gzip')
            f_out.create_dataset('labels', data=aug_labels, compression='gzip')
        print(f"Success! Saved {len(aug_images)} samples.")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()

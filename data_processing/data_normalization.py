import numpy as np
import h5py
import torch 
import os

'''This script focuses on normalizing the dataset using either Z-score or Min-Max normalization. It can handle both .h5 and .pt files as input and outputs normalized data in both formats. 
The normalization is done per image, treating each image independently to preserve relative differences within each sample.'''

def normalize_tensor(tensor, method="zscore"):
    tensor = tensor.astype(np.float32)
    reduce_axes = (1, 2) 

    if method.lower() == "minmax":
        t_min = tensor.min(axis=reduce_axes, keepdims=True)
        t_max = tensor.max(axis=reduce_axes, keepdims=True)
        normalized = (tensor - t_min) / (t_max - t_min + 1e-8)
    elif method.lower() == "zscore":
        mean = tensor.mean(axis=reduce_axes, keepdims=True)
        std = tensor.std(axis=reduce_axes, keepdims=True)
        normalized = (tensor - mean) / (std + 1e-8)
    else:
        raise ValueError("method must be 'minmax' or 'zscore'")

    return normalized

def main():
    # You can change this to either a .pt or .h5 file
    input_name = 'lfw_full_norm_subset_10.pt' 
    pt_output = 'lfw_full_norm_aug3_10.pt'
    output_name = 'lfw_full_norm_aug3_10.h5'
    
    print(f"Loading {input_name}...")
    
    images, labels = None, None
    extension = os.path.splitext(input_name)[1].lower()

    try:
        if extension == '.h5':
            with h5py.File(input_name, 'r') as f_in:
                # Using .get() helps avoid KeyErrors if names differ slightly
                images = f_in['images'][:] 
                labels = f_in['names'][:].squeeze() if 'names' in f_in else f_in['labels'][:].squeeze()
                
        elif extension == '.pt':
            data = torch.load(input_name, map_location='cpu')
            # Handles if .pt is a dictionary or a raw tensor
            if isinstance(data, dict):
                images = data['images'].numpy() if torch.is_tensor(data['images']) else data['images']
                labels = data['labels'].numpy() if torch.is_tensor(data['labels']) else data['labels']
            else:
                print("Note: .pt file loaded as raw data. Ensure it contains the expected structure.")
                images = data.numpy() if torch.is_tensor(data) else data
        else:
            print(f"Unsupported file extension: {extension}")
            return

    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if images is not None:
        print("Original dataset shape:", images.shape, labels.shape)

        # Step 1: Normalize
        normalize_method = 'zscore' 
        normalized_images = normalize_tensor(images, method=normalize_method)
        print("Normalization complete.")

        # Step 2: Save to HDF5 (Output remains H5 for consistency)
        output_name = f'LFW_full_{normalize_method}.h5'
        print(f"Saving to {output_name}...")

        
    print(f"Saving PyTorch to {pt_output}...")
    torch.save({
        'images': torch.from_numpy(normalized_images) if isinstance(normalized_images, np.ndarray) else normalized_images,
        'labels': torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels
    }, pt_output)


if __name__ == "__main__":
    main()

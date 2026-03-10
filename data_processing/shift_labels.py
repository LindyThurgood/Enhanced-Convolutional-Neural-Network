import h5py
import numpy as np
import torch
import os
'''This script loads two datasets (in .pt or .h5 format), shifts the labels of the second dataset to ensure no overlap with the first, and saves the adjusted second dataset to a new file. 
This was used to ensure label consistency when combining the two subsets of LFW. It was utilized before training the teacher model on the group of individuals with more than 10 images.'''
file1 = 'lfw_full_norm_augmented_9.pt'
file2 = 'lfw_full_norm_aug3_10.pt'
output = 'lfw_full_norm_aug3_10_shifted.pt'
def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.h5':
        with h5py.File(file_path, 'r') as f:
            return np.array(f['images'][:]), np.array(f['labels'][:])
    elif ext == '.pt':
        data = torch.load(file_path)
        return data['images'], data['labels']
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def shift_labels(source_file1, source_file2, output_file):
    _, labels1 = load_data(source_file1)
    max_label1 = labels1.max()
    images2, labels2 = load_data(source_file2)
    
    # Calculate the shift
    shift_val = max_label1 + 1
    shifted_labels2 = labels2 + shift_val
    
    print(f"Max label in {source_file1}: {max_label1}")
    print(f"Labels in {source_file2} shifted by: {shift_val}")
    
    print(f"New range: {shifted_labels2.min()} to {shifted_labels2.max()}")
    out_ext = os.path.splitext(output_file)[1].lower()
    
    if out_ext == '.h5':
        with h5py.File(output_file, 'w') as f_out:
            out_imgs = images2.numpy() if torch.is_tensor(images2) else images2
            out_lbls = shifted_labels2.numpy() if torch.is_tensor(shifted_labels2) else shifted_labels2
            f_out.create_dataset('images', data=out_imgs, compression="gzip")
            f_out.create_dataset('labels', data=out_lbls)
    elif out_ext == '.pt':
        torch_data = {
            'images': torch.as_tensor(images2),
            'labels': torch.as_tensor(shifted_labels2)
        }
        torch.save(torch_data, output_file)
            
    print(f"Successfully saved adjusted dataset to {output_file}")

shift_labels(file1, file2, output)

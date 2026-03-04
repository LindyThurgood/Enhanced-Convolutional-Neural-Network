import numpy as np
import h5py
import torch
from collections import Counter
'''This script creates two subsets of the LFW dataset based on the number of images per person.'''
# Load the dataset from HDF5 file
with h5py.File('lfw_dataset.h5', 'r') as f:
    images = np.array(f['images'])  #be sure to adjust key names if necessary
    names = np.array(f['names']).squeeze() 

print(f"Original dataset shape: {images.shape}")
print(f"Unique people: {len(np.unique(names))}")

# === 2. Count images per person ===
counts = Counter(names)

# Separate IDs based on image count
less_than_10_ids = [pid for pid, cnt in counts.items() if cnt < 10]
greater_equal_10_ids = [pid for pid, cnt in counts.items() if cnt >= 10]

#Filter datasets
def filter_subset(target_ids):
    mask = np.isin(names, target_ids)
    subset_images = images[mask]
    subset_names = names[mask]
    
    # Remap labels to contiguous 0..K-1
    unique_ids = np.unique(subset_names)
    id_map = {old: new for new, old in enumerate(unique_ids)}
    new_labels = np.array([id_map[i] for i in subset_names], dtype=int)
    
    return subset_images, new_labels

images_lt10, labels_lt10 = filter_subset(less_than_10_ids)
images_ge10, labels_ge10 = filter_subset(greater_equal_10_ids)

# Save to subsets to pytorch files
'''torch.save({
    'images': torch.from_numpy(images_lt10),
    'labels': torch.from_numpy(labels_lt10)
}, 'lfw_full_norm_subset_9.pt')'''

torch.save({
    'images': torch.from_numpy(images_ge10),
    'labels': torch.from_numpy(labels_ge10)
}, 'lfw_subset_10.pt')

print("Filtering complete")
print(f"Saved <10 subset: {images_lt10.shape[0]} images, {len(less_than_10_ids)} people.")
print(f"Saved >=10 subset: {images_ge10.shape[0]} images, {len(greater_equal_10_ids)} people.")

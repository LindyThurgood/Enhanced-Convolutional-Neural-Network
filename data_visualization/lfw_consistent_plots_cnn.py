import h5py
import umap
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

'''This script is created based on consistent_plots_CNN.py. It differs from that script in that it is designed
to display only a portion of the classes in the dataset. It utilizes an array to determine which portion to display.'''
def visualize_cnn_embeddings(embeddings, labels, dataset_name, title, subset_path=None, selected_classes=None):
    """
    Visualizes a subset of classes using UMAP with styling consistent with the rest of the project.
    """
    # 1. STYLE SETTINGS
    plt.style.use('seaborn-v0_8-muted') 
    plt.rcParams.update({
        "text.usetex": False,            
        "font.family": "serif",
        "axes.labelsize": 8,            
        "axes.titlesize": 9,            
        "xtick.labelsize": 7,           
        "ytick.labelsize": 7,
        "figure.dpi": 200                
    })

    # 2. DATA PREP
    data = embeddings.numpy() if hasattr(embeddings, 'numpy') else embeddings
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)

    all_unique_labels = np.unique(labels)

    # 3. CLASS SELECTION LOGIC
    # Priority: 1. Passed array -> 2. Saved File -> 3. Random Sample
    if selected_classes is not None:
        target_classes = np.sort(selected_classes)
        print(f"Using provided selection of {len(target_classes)} classes.")
    elif subset_path is not None and os.path.exists(subset_path):
        target_classes = np.sort(np.load(subset_path))
        print(f"Successfully loaded {len(target_classes)} classes from {subset_path}")
    else:
        print("No existing subset found. Selecting 100 random classes...")
        num_to_sample = min(100, len(all_unique_labels))
        target_classes = np.random.choice(all_unique_labels, size=num_to_sample, replace=False)
        target_classes = np.sort(target_classes)
        # Save for future use
        save_filename = f"{dataset_name}_subset.npy"
        np.save(save_filename, target_classes)
        print(f"Saved new subset to {save_filename}")

    # 4. FILTER DATA (Crucial for performance)
    mask_subset = np.isin(labels, target_classes)
    filtered_data = data[mask_subset]
    filtered_labels = labels[mask_subset]

    # 5. UMAP REDUCTION
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(filtered_data)

    # 6. PLOTTING
    plt.figure(figsize=(9, 6))
    colors = plt.cm.get_cmap('hsv', len(target_classes))

    for i, label in enumerate(target_classes):
        mask = (filtered_labels == label)
        if np.any(mask):
            plt.scatter(
                embedding_2d[mask, 0], 
                embedding_2d[mask, 1], 
                label=f'ID {int(label)}',
                color=colors(i),
                alpha=0.8,
                s=12,
                edgecolors='black',
                linewidth=0.2
            )

    plt.title(title, loc='left', pad=10)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.legend(
        loc='upper left', 
        bbox_to_anchor=(1, 1), 
        fontsize=4, 
        frameon=False, 
        ncol=2, 
        title="Classes",
        title_fontsize=6
    )
    
    plt.tight_layout()
    plt.show()
    return target_classes

def load_data(file_path):
    """
    Detects file type and returns (embeddings, labels) as numpy arrays.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # --- HANDLE PYTORCH FILES ---
    if ext == '.pt':
        data_dict = torch.load(file_path, map_location='cpu', weights_only=False)
        # Handle case where file is a dictionary
        if isinstance(data_dict, dict):
            img_key = next((k for k in ['embeddings', 'images', 'data'] if k in data_dict), None)
            lbl_key = next((k for k in ['labels', 'targets', 'target', 'names'] if k in data_dict), None)
            
            if not img_key or not lbl_key:
                raise KeyError(f"Keys not found in .pt file. Found: {list(data_dict.keys())}")
            
            embeddings = data_dict[img_key]
            labels = data_dict[lbl_key]
        # Handle case where file is a list/tuple: [embeddings, labels]
        elif isinstance(data_dict, (list, tuple)):
            embeddings, labels = data_dict[0], data_dict[1]
        
        # Convert Tensors to Numpy
        if torch.is_tensor(embeddings): embeddings = embeddings.detach().numpy()
        if torch.is_tensor(labels): labels = labels.detach().numpy()

    # --- HANDLE H5 FILES ---
    elif ext == '.h5':
        with h5py.File(file_path, 'r') as f:
            img_key = next((k for k in ['images', 'embeddings', 'data'] if k in f), None)
            lbl_key = next((k for k in ['labels', 'names', 'target'] if k in f), None)
            
            if not img_key or not lbl_key:
                raise KeyError(f"Keys not found in .h5 file. Found: {list(f.keys())}")
            
            embeddings = f[img_key][:]
            labels = f[lbl_key][:].flatten()

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return np.array(embeddings), np.array(labels).flatten()

def main():
    # SET YOUR VARIABLE HERE
    # input_file = 'lfw_dataset.h5' 
    #input_file = 'Full_lfw_preprocessed.pt'
    input_file = 'lfw_plain_CNN_embeddings.h5'
    subset_file = 'LFW_subset.npy'
    
    try:
        if not os.path.exists(input_file):
            print(f"File {input_file} not found.")
            return

        print(f"Attempting to load: {input_file}")
        embeddings, labels = load_data(input_file)
        print(f"Successfully loaded {len(labels)} samples.")

        # Run visualization
        visualize_cnn_embeddings(
            embeddings, 
            labels, 
            "LFW_Dataset", 
            f"LFW Plain CNN Embeddings UMAP Visualization", 
            subset_path=subset_file
        )

    except Exception as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    main()

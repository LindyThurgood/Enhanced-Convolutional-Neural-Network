import h5py 
import umap
import numpy as np
import matplotlib.pyplot as plt
'''This script is designed to visualize the CNN-derived embeddings and the raw data using UMAP. 
Different color schemes are applied to distinguish between classes, and the visualizations are styled for clarity and consistency with the project.'''
def visualize_cnn_embeddings(embeddings, labels, dataset_name, title):
    """
    Refined UMAP visualization with a top-aligned, borderless legend.
    """
    # 1. STYLE SETTINGS: Professional and minimalist
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
    if hasattr(embeddings, 'numpy'):
        data = embeddings.numpy()
    else:
        data = embeddings
    
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)

    # 3. UMAP Reduction
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(data)

    # 4. PLOTTING
    plt.figure(figsize=(6, 4))
    
    unique_labels = np.sort(np.unique(labels))
    # 'Set1' for high contrast between 0 and 1
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    #tab10 for Olivetti, Set1 for Abide

    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        plt.scatter(
            embedding_2d[mask, 0], 
            embedding_2d[mask, 1], 
            label=f'{int(label)}',
            color=colors(i),
            alpha=0.8,
            s=12,
            edgecolors='black',
            linewidth=0.2
        )

    plt.title(title, loc='left', pad=10) # 'left' alignment for a modern look
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    # Remove outer boundaries for a cleaner aesthetic
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 5. TOP-ALIGNED BORDERLESS LEGEND
    plt.legend(
        loc='upper left',        # Alignment point of the legend itself
        bbox_to_anchor=(1, 1),   # Coordinates relative to the chart (1, 1 is top-right)
        fontsize=4, 
        frameon=False,           # No boundary box
        title="Class",
        title_fontsize=6
    )
    
    plt.tight_layout()
    plt.show()

def main():
    # Specific file from your ABIDE project work
    file_path = 'Olivetti_plain_CNN_embeddings.h5'
    try:
        with h5py.File(file_path, 'r') as f:
            embeddings = f['embeddings'][:]
            labels = f['labels'][:].flatten()
        
        visualize_cnn_embeddings(embeddings, labels, "Olivetti", "Olivetti CNN Feature Space UMAP Visualization")
    except Exception as e:
        print(f"Error: {e}")

# Path to your raw data H5 file
    file_path = 'olivetti_data.h5' 
    
    try:
        with h5py.File(file_path, 'r') as f:
            # CHANGE THESE KEYS to match your raw data H5 structure
            # Use f.keys() to print and verify available datasets
            raw_data = f['images'][:] 
            labels = f['labels'][:].flatten()
        
        visualize_cnn_embeddings(
            raw_data, 
            labels, 
            "Olivetti Raw", 
            "UMAP Visualization of Raw Olivetti Data"
        )
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

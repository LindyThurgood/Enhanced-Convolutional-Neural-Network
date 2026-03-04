from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import umap
import h5py
'''This script is designed to compare the clustering performance of different dimensionality reduction techniques (PCA, t-SNE, UMAP) and CNN-derived embeddings on a given dataset. 
It includes a refined visualization with integrated titles, legends, and consistent styling for better interpretability. 
The script loads data from an HDF5 file, processes it, and generates a comprehensive comparison of clustering methods using silhouette scores, Davies-Bouldin scores, Calinski-Harabasz scores as well as visualizations.
This code was obtained for Maryam Bagharian and further refined to be consistent with the image styling of the project.'''
def compare_clustering_methods(original_data, labels, metric_embeddings, dataset_name):
    """Refined comparison with Component labels, title-key, and unified bar colors."""
    
    # 1. STYLE SETTINGS
    plt.rcParams.update({
        "text.usetex": False,            
        "font.family": "serif",
        "axes.labelsize": 8,            
        "axes.titlesize": 9,            
        "xtick.labelsize": 7,           
        "ytick.labelsize": 7,
        "figure.dpi": 200                
    })

    # Prepare data
    original_flat = original_data.reshape(original_data.shape[0], -1)
    labels = labels.flatten()
    unique_labels = np.sort(np.unique(labels))
    
    # Define High-Contrast Colors
    colors = plt.cm.get_cmap('Set1', len(unique_labels))
    class_1_color = colors(1) # Color for Class 1

    methods = {
        'Raw Data': original_flat,
        'PCA': PCA(n_components=20).fit_transform(original_flat),
        't-SNE': TSNE(n_components=2, random_state=42).fit_transform(original_flat),
        'UMAP': umap.UMAP(random_state=42).fit_transform(original_flat),
        'CNN Features': metric_embeddings
    }
    
    results = []
    processed_data = {}

    for method_name, data in methods.items():
        kmeans = KMeans(n_clusters=len(unique_labels), random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(data)
        results.append({'Method': method_name, 'Silhouette': silhouette_score(data, clusters)})
        processed_data[method_name] = data
    
    # 2. VISUALIZATION
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    # Plot Scatter Graphs
    for i, (method_name, data) in enumerate(list(processed_data.items())):
        ax = axes[i//3, i%3]
        viz_data = PCA(n_components=2).fit_transform(data) if data.shape[1] > 2 else data
            
        for j, label in enumerate(unique_labels):
            mask = (labels == label)
            ax.scatter(viz_data[mask, 0], viz_data[mask, 1], 
                       color=colors(j), alpha=0.7, s=15, 
                       edgecolors='black', linewidth=0.2)
        
        ax.set_title(method_name, loc='left')
        
        # RESTORED LABELS
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 3. METRICS BAR CHART (Unified Color)
    ax_bar = axes[1, 2]
    ax_bar.bar(range(len(results)), [r['Silhouette'] for r in results], 
               color=class_1_color, alpha=0.8) 
    ax_bar.set_title('Silhouette Comparison', loc='left')
    ax_bar.set_xticks(range(len(results)))
    ax_bar.set_xticklabels([r['Method'] for r in results], rotation=30, ha='right')
    ax_bar.set_ylabel('Score')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    
    # 4. INTEGRATED TITLE AND KEY
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='0',
                              markerfacecolor=colors(0), markersize=8),
                       Line2D([0], [0], marker='o', color='w', label='1',
                              markerfacecolor=colors(1), markersize=8)]
    
    fig.suptitle(f'Clustering Comparison: {dataset_name}', fontsize=12, fontweight='bold', x=0.08, y=0.98, ha='left')
    
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.98), 
               ncol=2, frameon=False, title="Class", fontsize=9, title_fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f'clustering_comparison_{dataset_name}.jpg', dpi=300)
    plt.show()

    return pd.DataFrame(results)

def main():
    try:
        with h5py.File('cnn_ed_Abide_DNA_BEST_output.h5', 'r') as data:
            embeddings = data['embeddings'][:] 
            labels = data['labels'][:]  
        print(f"Original Shape of X: {embeddings.shape}")
    except OSError:
        print("Error: File 'cnn_ed_Abide_DNA_BEST_output.h5' not found.")
        exit()
    dataset_name = 'ABIDE'
    mat = loadmat('AugNormAbide.mat')
    #original_data =load_and_concatenate_chunks()
    original_data = mat['normalized_matrices'] 
    '''with h5py.File('lfw_9_test_augmented_zscore.h5', 'r') as data:
            original_data = data['images'][:]   
    print(f"Original Shape of data: {original_data.shape}")'''
    
    results = compare_clustering_methods(original_data, labels, embeddings, dataset_name)
    print(results)

if __name__ == "__main__":
    main()

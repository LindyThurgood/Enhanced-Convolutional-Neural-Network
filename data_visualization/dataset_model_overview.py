import h5py
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from matplotlib.lines import Line2D

'''This script allows for mutliple models/views of the data to be compared side by side. It's adapted from compare_clustering.py to have the same visualization 
layout for consistency.'''

def compare_make_plots_umap_final(dataset_name):
    
    # Style settings 
    plt.rcParams.update({
        "text.usetex": False,            
        "font.family": "serif",
        "axes.labelsize": 8,            
        "axes.titlesize": 9,            
        "xtick.labelsize": 7,           
        "ytick.labelsize": 7,
        "figure.dpi": 200                
    })

    methods = {#Model/data title : file path
        'Raw Data': 'Abide_raw.h5',
        'Preprocessed Data': 'AugNormAbide.h5',
        'Abide Model 1': 'Abide_raw_CNN_embeddings.h5',
        'Abide Model 2': 'AugNormAbide_plain_CNN_embeddings.h5',
        'Abide Model 3': 'AugNormAbide_CNN_ED_embeddings.h5',
        'Abide Model 4': 'KD_AugNormAbide_T3A05_embeddings.h5'
    }
    
    results = []
    processed_plots = {}
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

    for method_name, file_path in methods.items():
        try:
            with h5py.File(file_path, 'r') as f:
                if 'embeddings' in f:
                    data = f['embeddings'][:]
                elif 'normalized_matrices' in f:
                    data = f['normalized_matrices'][:]
                elif 'images' in f:
                    data = f['images'][:]
                else:
                    print(f"Warning: No valid data key found in {file_path}. Keys: {list(f.keys())}")
                    continue
                
                current_labels = f['labels'][:].flatten()
            
            if len(data.shape) > 2:
                data = data.reshape(data.shape[0], -1)

            # Calculate Metrics
            sil = silhouette_score(data, current_labels)
            dbi = davies_bouldin_score(data, current_labels)
            chi = calinski_harabasz_score(data, current_labels)
            
            display_name = "Data" if method_name == "Preprocessed Data" else method_name
            
            results.append({
                'Method': display_name, 
                'Silhouette': sil,
                'Davies-Bouldin': dbi,
                'Calinski-Harabasz': chi
            })
            
            # DIMENSIONALITY REDUCTION
            print(f"Processing UMAP for: {method_name}...")
            embedding_2d = reducer.fit_transform(data)
            processed_plots[method_name] = (embedding_2d, current_labels, sil)

        except Exception as e:
            print(f"Error processing {method_name} ({file_path}): {e}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    
    unique_labels = [0, 1] # Currently Set up for ABIDE I adjust if other classes needed
    colors = plt.cm.get_cmap('Set1', 2)

    for i, (method_name, (viz_data, cur_labels, score)) in enumerate(processed_plots.items()):
        ax = axes_flat[i]
            
        for j, label in enumerate(unique_labels):
            mask = (cur_labels == label)
            ax.scatter(viz_data[mask, 0], viz_data[mask, 1], 
                       color=colors(j), alpha=0.7, s=15, 
                       edgecolors='black', linewidth=0.2)
        
        display_title =  method_name
        ax.set_title(f"{display_title}", loc='left')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend and Title
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='ASD', markerfacecolor=colors(0), markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Control', markerfacecolor=colors(1), markersize=8)
    ]
    
    fig.suptitle(f'CNN Model Comparison: ABIDE I', 
                 fontsize=14, fontweight='bold', x=0.05, y=0.98, ha='left')
    
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.98), 
               ncol=2, frameon=False, fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    # Save output
    output_filename = f'umap_clustering_{dataset_name}.jpg'
    fig.savefig(output_filename, dpi=300)
    print(f"Visualization saved as {output_filename}")
    plt.show()

    return pd.DataFrame(results)

if __name__ == "__main__":
    dataset_name = 'ABIDE'
    metrics_df = compare_make_plots_umap_final(dataset_name)
    
    print("\n--- Final Performance Metrics ---")
    print(metrics_df.to_string(index=False))

import numpy as np
import h5py
import joblib
import matplotlib.pyplot as plt
import umap
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                             silhouette_score, davies_bouldin_score, 
                             calinski_harabasz_score)

'''This script creates an SVM classifier using the feature vectors extracted from a CNN. It evaluates the 
classification performance using accuracy, precision, recall, and F1-score, as well as clustering metrics 
like silhouette score, Davies-Bouldin index, and Calinski-Harabasz index. Finally, it visualizes the 
SVM-partitioned space using UMAP for dimensionality reduction.'''

def evaluate_and_visualize_clean(embeddings, labels):
    plt.style.use('seaborn-v0_8-muted') 
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 8,            
        "axes.titlesize": 9,            
        "figure.dpi": 200                
    })
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    sil_svm = silhouette_score(X_test, y_pred)
    db_svm = davies_bouldin_score(X_test, y_pred)
    ch_svm = calinski_harabasz_score(X_test, y_pred)

    #pring report
    print("\n" + "="*50)
    print(f"SVM CLASSIFICATION & CLUSTERING REPORT")
    print("-" * 50)
    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print("-" * 50)
    print(f"Silhouette (↑):      {sil_svm:.4f}")
    print(f"Davies-Bouldin (↓):  {db_svm:.4f}")
    print(f"Calinski-Harab. (↑): {ch_svm:.4f}")
    print("="*50)

    # UMAP REDUCTION
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(X_test)

    #PLOTTING
    plt.figure(figsize=(7, 4))
    unique_preds = np.sort(np.unique(y_pred))
    colors = plt.cm.get_cmap('Set1', len(unique_preds))
    #tab10 for Olivetti, Set1 for Abide

    for i, label in enumerate(unique_preds):
        mask = (y_pred == label)
        plt.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1], 
            label=f'Class {int(label)}', color=colors(i),
            alpha=0.8, s=15, edgecolors='black', linewidth=0.2
        )

    plt.title(f"SVM Partitioned Space", loc='left', pad=10)
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
        title="Class",
        title_fontsize=6
    )
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'cnn_ed_Abide_DNA_BEST_output.h5'
    try:
        with h5py.File(file_path, 'r') as f:
            embeddings = f['embeddings'][:]
            labels = f['labels'][:].flatten()
        evaluate_and_visualize_clean(embeddings, labels)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")

if __name__ == "__main__":
    main()

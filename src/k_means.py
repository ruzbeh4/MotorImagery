import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def evaluate_kmeans_k(X, max_k=10):
    k_values = range(2, max_k + 1)
    wcss = []
    silhouette_scores = []
    
    for k in k_values:
        # 10 different centroid seeds
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        wcss.append(kmeans.inertia_) 
        silhouette_scores.append(silhouette_score(X, labels))
        
    return list(k_values), wcss, silhouette_scores

def plot_kmeans_metrics(k_values, wcss, silhouette_scores):
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Plot WCSS ->  left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('WCSS (Inertia)', color=color, fontweight='bold')
    ax1.plot(k_values, wcss, marker='o', color=color, linewidth=2, label='WCSS')
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Silhouette -> right y-axis
    ax2 = ax1.twinx()      
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color, fontweight='bold')
    ax2.plot(k_values, silhouette_scores, marker='s', color=color, linewidth=2, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Determining Optimal k: WCSS vs Silhouette Score')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_kmeans_scatter(X_2d, true_labels, cluster_labels, optimal_k):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Truth
    scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=true_labels, cmap='coolwarm', edgecolors='k', s=60)
    axes[0].set_title('Ground Truth (Actual Labels)')
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    
    # K-Means Result
    scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap='viridis', edgecolors='k', s=60)
    axes[1].set_title(f'K-Means Clusters (k={optimal_k})')
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    
    plt.suptitle('Evaluating CSP Feature Separability via Unsupervised Clustering')
    plt.show()
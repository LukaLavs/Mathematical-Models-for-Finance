import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def compute_correlation_matrix(Sigma):
    """Compute the correlation matrix from a covariance matrix."""
    D = np.sqrt(np.diag(Sigma))
    Corr = Sigma / np.outer(D, D)
    np.fill_diagonal(Corr, 1.0)
    return Corr


def cluster_assets(corr_matrix, num_clusters, use_abs=True):
    """Cluster assets using hierarchical clustering on correlation matrix."""
    if use_abs:
        corr_matrix = np.abs(corr_matrix)
    # Convert correlation to distance
    dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method='ward')
    labels = fcluster(Z, num_clusters, criterion='maxclust')
    return labels


def hierarchical_optimization(model, S, m=None, delta=0.5, num_clusters=3, use_abs_corr=True):
    """Full pipeline for hierarchical portfolio optimization."""
    N = S.shape[0]
        
    # Step 1: Compute correlation matrix
    corr_matrix = compute_correlation_matrix(S)

    # Step 2: Cluster assets
    cluster_labels = cluster_assets(corr_matrix, num_clusters, use_abs_corr)

    # Step 3: Intra-cluster optimization
    cluster_weights = []
    cluster_indices = []
    for k in range(1, num_clusters + 1):
        indices = np.where(cluster_labels == k)[0]
        S_k = S[np.ix_(indices, indices)]
        m_k = m[indices] if m is not None else None

        S_k = np.atleast_2d(S_k) # Ensures dimensions in case 1X1
        w_k, _, _ = model(S_k, m_k, delta)
        cluster_weights.append(w_k)
        cluster_indices.append(indices)

    # Step 4: Build reduced covariance matrix (inter-cluster)
    reduced_cov = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(num_clusters):
            idx_i = cluster_indices[i]
            idx_j = cluster_indices[j]
            w_i = cluster_weights[i]
            w_j = cluster_weights[j]
            S_ij = S[np.ix_(idx_i, idx_j)]
            reduced_cov[i, j] = w_i @ S_ij @ w_j

    # Step 5: Build reduced expected return vector
    if m is not None:
        reduced_m = np.array([
            cluster_weights[i] @ m[cluster_indices[i]]
            for i in range(num_clusters)
        ])
    else:
        reduced_m = None

    # Step 6: Inter-cluster optimization
    cluster_level_weights, _, _ = model(reduced_cov, reduced_m, delta)

    # Step 7: Compose final portfolio
    final_weights = np.zeros(N)
    for k in range(num_clusters):
        idx = cluster_indices[k]
        final_weights[idx] = cluster_level_weights[k] * cluster_weights[k]
    
    returns = final_weights.T @ m
    risk = np.sqrt(final_weights @ S @ final_weights)

    # Could return cluster_labels as well
    return final_weights, returns, risk






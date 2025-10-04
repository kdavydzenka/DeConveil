import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import random

import matplotlib.pyplot as plt
import seaborn as sns


def pca_cluster_cn(
    gene_cn: pd.DataFrame,
    n_components: int = 20,
    k: int = 2,
    method: str = "kmeans",
    random_state: int = 0,
) -> dict:
    """
    Perform PCA on gene-level CN and cluster patients in PCA space.

    Parameters
    ----------
    gene_cn : DataFrame
        Gene x Sample matrix of CN values (log2 ratios).
    n_components : int
        Number of PCA components to keep.
    k : int
        Number of clusters.
    method : str
        'kmeans' or 'hierarchical'.
    random_state : int
        For reproducibility.

    Returns
    -------
    dict with:
        - labels: pd.Series (sample -> cluster)
        - pca_coords: DataFrame of PCA coords
        - explained_var: explained variance ratios
    """
    X = gene_cn.fillna(0).T  # samples × genes
    pca = PCA(n_components=min(n_components, X.shape[1]))
    coords = pca.fit_transform(X)
    coords_df = pd.DataFrame(
        coords, index=X.index, columns=[f"PC{i+1}" for i in range(coords.shape[1])]
    )

    if method == "kmeans":
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(coords)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(coords)
    else:
        raise ValueError("method must be 'kmeans' or 'hierarchical'")

    labels = pd.Series(labels, index=X.index, name="cluster")
    return {
        "labels": labels,
        "pca_coords": coords_df,
        "explained_var": pca.explained_variance_ratio_,
    }


def consensus_cluster_cn(
    gene_cn: pd.DataFrame,
    k: int = 2,
    n_resamples: int = 50,
    sample_fraction: float = 0.8,
    feature_fraction: float = 0.8,
    top_genes: int = 2000,
    random_state: int = 0,
) -> dict:
    """
    Consensus clustering of patients based on CN profiles.

    Parameters
    ----------
    gene_cn : DataFrame
        Gene x Sample CN matrix.
    k : int
        Number of clusters.
    n_resamples : int
        Number of resampling iterations.
    sample_fraction : float
        Fraction of patients sampled each iteration.
    feature_fraction : float
        Fraction of genes sampled each iteration.
    top_genes : int
        Use top variable genes only.
    random_state : int
        For reproducibility.

    Returns
    -------
    dict with:
        - labels: pd.Series (sample -> cluster) from consensus
        - consensus_matrix: DataFrame (samples × samples) with co-clustering frequencies
    """
    rng = np.random.RandomState(random_state)

    # Select top variable genes
    var_genes = gene_cn.var(axis=1).sort_values(ascending=False).index[:top_genes]
    data = gene_cn.loc[var_genes].fillna(0).values  # genes × samples
    samples = gene_cn.columns.tolist()
    n = len(samples)

    co_mat = np.zeros((n, n))
    counts = np.zeros((n, n))

    for r in range(n_resamples):
        samp_idx = rng.choice(n, size=int(sample_fraction * n), replace=False)
        feat_idx = rng.choice(
            data.shape[0], size=int(feature_fraction * data.shape[0]), replace=False
        )
        X = data[feat_idx][:, samp_idx].T  # subsampled patients × genes

        # k-means in subsample
        km = KMeans(n_clusters=k, n_init=10, random_state=rng).fit(X)
        labels_sub = km.labels_

        # update co-occurrence
        for i, si in enumerate(samp_idx):
            for j, sj in enumerate(samp_idx):
                counts[si, sj] += 1
                if labels_sub[i] == labels_sub[j]:
                    co_mat[si, sj] += 1

    consensus = np.divide(co_mat, counts, out=np.zeros_like(co_mat), where=counts > 0)
    consensus_df = pd.DataFrame(consensus, index=samples, columns=samples)

    # Cluster consensus matrix
    dist = 1 - consensus
    agg = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    labels = agg.fit_predict(dist)
    labels = pd.Series(labels, index=samples, name="cluster")

    return {"labels": labels, "consensus_matrix": consensus_df}


def consensus_cdf_range(
    gene_cn, k_values=(2,3,4,5,6),
    n_resamples=50, sample_fraction=0.8, feature_fraction=0.8,
    top_genes=2000, random_state=0
):
    """
    Run consensus clustering across multiple k and plot CDFs.

    Parameters
    ----------
    gene_cn : DataFrame
        Gene × Sample CN matrix.
    k_values : list/tuple
        Range of k to test.
    n_resamples, sample_fraction, feature_fraction, top_genes, random_state
        Passed to consensus_cluster_cn().

    Returns
    -------
    dict
        {k: {"labels", "consensus_matrix", "auc"}}
    """
    results = {}

    plt.figure(figsize=(7,5))

    for k in k_values:
        res = consensus_cluster_cn(
            gene_cn, k=k,
            n_resamples=n_resamples,
            sample_fraction=sample_fraction,
            feature_fraction=feature_fraction,
            top_genes=top_genes,
            random_state=random_state
        )

        mat = res["consensus_matrix"].values
        mask = ~np.eye(mat.shape[0], dtype=bool)
        vals = mat[mask]

        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)

        # Compute area under CDF (AUC)
        auc = np.trapz(cdf, sorted_vals)
        res["auc"] = auc
        results[k] = res

        plt.plot(sorted_vals, cdf, lw=2, label=f"k={k} (AUC={auc:.3f})")

    plt.xlabel("Consensus value")
    plt.ylabel("Cumulative fraction")
    plt.title("Consensus CDF across k", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return results
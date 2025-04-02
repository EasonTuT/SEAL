import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             normalized_mutual_info_score, silhouette_score)

from metric import cluster_acc

def cluster_embedding(embedding, cluster_number, Y, save_pred=False,
                      leiden_n_neighbors=300, cluster_methods=["KMeans", "Leiden"]):
    """[summary]

    Args:
        embedding ([type]): [description]
        cluster_number ([type]): [description]
        Y ([type]): [description]
        save_pred (bool, optional): [description]. Defaults to False.
        leiden_n_neighbors (int, optional): [description]. Defaults to 300.

    Returns:
        [type]: [description]
    """
    result = {"t_clust": time.time()}
    if "KMeans" in cluster_methods:
        # evaluate K-Means
        kmeans = KMeans(n_clusters=cluster_number,
                        init="k-means++",
                        random_state=0)
        pred = kmeans.fit_predict(embedding)
        if Y is not None:
            result[f"kmeans_ari"] = adjusted_rand_score(Y, pred)
            result[f"kmeans_nmi"] = normalized_mutual_info_score(Y, pred)
        result[f"kmeans_sil"] = silhouette_score(embedding, pred)
        result[f"kmeans_cal"] = calinski_harabasz_score(embedding, pred)
        result[f"kmeans_acc"] = cluster_acc(Y,pred)
        result["t_k"] = time.time()
        if save_pred:
            result[f"kmeans_pred"] = pred

    # if "Leiden" in cluster_methods:
    #     # evaluate leiden
    #     pred = utils.run_leiden(embedding, leiden_n_neighbors)
    #     if Y is not None:
    #         result[f"leiden_ari"] = adjusted_rand_score(Y, pred)
    #         result[f"leiden_nmi"] = normalized_mutual_info_score(Y, pred)
    #     result[f"leiden_sil"] = silhouette_score(embedding, pred)
    #     result[f"leiden_cal"] = calinski_harabasz_score(embedding, pred)
    #     result["t_l"] = time.time()
    #     if save_pred:
    #         result[f"leiden_pred"] = pred

    return result

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size



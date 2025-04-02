import math

import numpy as np
from sklearn.utils import shuffle


def next_batch(x, batch_size):
    num_samples = x.shape[0]
    index = np.linspace(0, num_samples - 1, num_samples, dtype=int)
    index = shuffle(index)
    # total = int(math.ceil(num_samples / batch_size))

    total = int(math.floor(num_samples / batch_size))

    for i in range(total):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(num_samples, end_idx)
        idx = index[start_idx: end_idx]
        batch_x = x[idx]
        yield (batch_x, (i + 1))


import warnings

import scipy.io as sio
import numpy as np
import torch
import pandas as pd


import scanpy as sc
import scipy as sp

data_names = {
    "human_ESC": "human_ESC"
}


def normalize(adata, copy=True, highly_genes=None, filter_min_counts=True,
              size_factors=True, normalize_input=True, logtrans_input=True):
    """
    Normalizes input data and retains only most variable genes
    (indicated by highly_genes parameter)

    Args:
        adata ([type]): [description]
        copy (bool, optional): [description]. Defaults to True.
        highly_genes ([type], optional): [description]. Defaults to None.
        filter_min_counts (bool, optional): [description]. Defaults to True.
        size_factors (bool, optional): [description]. Defaults to True.
        normalize_input (bool, optional): [description]. Defaults to True.
        logtrans_input (bool, optional): [description]. Defaults to True.

    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    # print("adata X shape:", adata.X.shape)  # [cell_num, features]
    if adata.X.size < 50e6:  # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error
    # 仅在一个或更少细胞中表达的基因被丢弃
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)  # 3

        sc.pp.filter_cells(adata, min_counts=1)
        # print("adata X shape:", adata.X.shape)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:  ## error
        # sc.pp.normalize_per_cell(adata)
        sc.pp.normalize_total(adata)  # 对表达计数矩阵进行归一化
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        # min_mean / max_mean -> 筛选基因的最大均值 ； min_disp -> 筛选基因的最小方差值
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes,
                                    subset=True)  # 根据其分散排名选择最可变的基因(前500个基因)
    if normalize_input:
        sc.pp.scale(adata)  # 对数据进行缩放
    return adata


def preprocess(X, nb_genes=500):
    """
    Preprocessing phase as proposed in scanpy package.
    Keeps only nb_genes most variable genes and normalizes
    the data to 0 mean and 1 std.
    Args:
        X ([type]): [description]
        nb_genes (int, optional): [description]. Defaults to 500.
    Returns:
        [type]: [description]
    """
    X = np.ceil(X).astype(int)
    count_X = X
    print(X.shape, count_X.shape, f"keeping {nb_genes} genes")
    orig_X = X.copy()
    adata = sc.AnnData(X)

    adata = normalize(adata,
                      copy=True,
                      highly_genes=nb_genes,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    X = adata.X.astype(np.float32)
    return X


def get_datamat(data_name, nb_genes, shape, device, process=True):
    if data_name[:4] == "pbmc":
        features = pd.read_csv("./datasets/pbmc_133.csv", index_col=0).values

        labels = pd.read_csv("./datasets/pbmc_labels.csv", index_col=0).values

        print("pbmc shape:", features.shape)
        print(labels.shape)
        ori_matrix = features
        features = features.reshape((-1, 1, shape, shape))

        # 采样
        if data_name == "pbmc10000":
            sample_num = 10000
        elif data_name == "pbmc5000":
            sample_num = 5000
        elif data_name == "pbmc2000":
            sample_num = 2000
        else:
            sample_num = 1000

        sample_idx = np.linspace(0, 10000, sample_num, endpoint=False, dtype=np.int32)
        ori_matrix = ori_matrix[sample_idx]
        features = features[sample_idx]
        labels = labels[sample_idx]
    elif data_name == "Macaque":
        data_path = "./Macaque/data.csv"
        label_path = "./Macaque/label.csv"
        features = pd.read_csv(data_path, header=0, index_col=0, sep=',')
        labels = pd.read_csv(label_path, index_col=0, header=0, sep=',')
        features = normalize(features, norm='l1', axis=1)
        features = features.reshape((-1, 1, shape, shape))
    else:
        filename = "./datasets/" + data_names[data_name] + ".mat"

        # 读取数据
        data = sio.loadmat(filename)
        # new
        features, labels = data['fea'], data['label']
        # print("original feat shape:", features.shape, labels.shape) # (num_sample, num_original_feats)

    # 数据处理
    features = np.where(features < 0, 0, features)
    print('before return', features.shape)
    # print("f1 shape:", features.shape)
    # print(features)
    if process:
        features = preprocess(features, nb_genes=nb_genes)

    features = torch.from_numpy(features).float().to(device)
    labels = np.squeeze(labels - np.min(labels))

    # print('before return', features.shape)  # [num_sample, 1, num_feats, num_feats]
    return features, labels


# 加载其他参数
def load_train_data(db, nb_genes, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), process=True):
    X, Y = get_datamat(data_name=db, nb_genes=nb_genes, shape=None, device=device, process=process)
    # X = X.T
    # X, Y, dims, data_size, class_num
    print(X.shape)
    dims = X.shape[0]
    data_size = X.shape[1]
    num_class = len(np.unique(Y))
    return X, Y, dims, data_size, num_class

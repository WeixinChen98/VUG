import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


def uniformity(x):
    x = F.normalize(x, dim=-1) 
    return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()


def alignment(x, y):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(2).mean()

def get_norm_adj_mat(interaction_matrix, n_users=None, n_items=None):
    # build adj matrix
    if n_users == None or n_items == None:
        n_users, n_items = interaction_matrix.shape
    A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    inter_M = interaction_matrix
    inter_M_t = interaction_matrix.transpose()
    data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
    data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
    A._update(data_dict)
    # norm adj matrix
    sumArr = (A > 0).sum(axis=1)
    # add epsilon to avoid divide by zero Warning
    diag = np.array(sumArr.flatten())[0] + 1e-7
    diag = np.power(diag, -0.5)
    D = sp.diags(diag)
    L = D * A * D
    # covert norm_adj matrix to tensor
    L = sp.coo_matrix(L)
    row = L.row
    col = L.col
    i = torch.LongTensor([row, col])
    data = torch.FloatTensor(L.data)
    SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
    return SparseL
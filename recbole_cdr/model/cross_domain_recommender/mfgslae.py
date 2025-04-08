import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from torch.distributions.binomial import Binomial

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, real_value_mask):
        binarized_mask = (real_value_mask >= 0).float().requires_grad_(True)
        ctx.save_for_backward(binarized_mask.clone())
        return binarized_mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class KernelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_factor,
                    nb_random_features=10, weight=True):
        super(KernelAttention, self).__init__()
        if weight:
            self.Wk = nn.ModuleList()
            self.Wq = nn.ModuleList()
            self.Wv = nn.ModuleList()
            for _ in range(num_factor):
                self.Wk.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))
                self.Wq.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))
                self.Wv.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))

        self.out_channels = out_channels
        self.num_factor = num_factor
        self.nb_random_features = nb_random_features
        self.weight = weight

    def reset_parameters(self):
        self.apply(xavier_normal_initialization)

    def forward(self, z, tau):
        query, key, value = torch.zeros_like(z, device=z.device), torch.zeros_like(z, device=z.device), torch.zeros_like(z, device=z.device)
        for head in range(self.num_factor):
            query[:, head] = self.Wq[head](z[:, head])
            key[:, head] = self.Wk[head](z[:, head])
            value[:, head] = self.Wv[head](z[:, head])

        dim = query.shape[-1]
        projection_matrix = create_projection_matrix(self.nb_random_features, dim).to(query.device)
        z_next = kernelized_softmax(query, key, value, projection_matrix, tau)

        z_next = z_next.flatten(-2, -1)
        return z_next.squeeze()

def adj_mul(adj_i, adj, N):
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()
    return adj_j

def create_projection_matrix(m, d):
    nb_full_blocks = int(m/d)
    block_list = []
    for _ in range(nb_full_blocks):
        unstructured_block = torch.randn((d, d))
        q, _ = torch.linalg.qr(unstructured_block)
        block_list.append(q.T)
    final_matrix = torch.vstack(block_list)
    multiplier = torch.norm(torch.randn((m, d)), dim=1)
    return torch.matmul(torch.diag(multiplier), final_matrix)

import math

def softmax_kernel(data, is_query, projection_matrix, eps=1e-4):
    data_normalizer = (data.shape[-1] ** -0.25)
    ratio = (projection_matrix.shape[0] ** -0.5)

    data_dash = torch.einsum("nhd,md->nhm", (data_normalizer * data), projection_matrix) # perform projection
    diag_data = (data ** 2).sum(-1)

    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(-1)
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps
        )
    return data_dash


def kernelized_softmax(query, key, value, projection_matrix=None, tau=0.5):
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)

    query_kernel = softmax_kernel(query, True, projection_matrix)
    key_kernel = softmax_kernel(key, False, projection_matrix)

    kvs = torch.einsum("nhm,nhd->hmd", key_kernel, value)
    numerator = torch.einsum("nhm,hmd->nhd", query_kernel, kvs)
    denominator = (query_kernel * key_kernel.sum(0, keepdim=True)).sum(-1, keepdim=True)

    z_output = numerator / denominator
    return z_output

class AutoEncoderMixin(object):
    """This is a common part of auto-encoders. All the auto-encoder models should inherit this class,
    including CDAE, MacridVAE, MultiDAE, MultiVAE, RaCT and RecVAE.
    The base AutoEncoderMixin class provides basic dataset information and rating matrix function.
    """

    def convert_sparse_matrix_to_rating_matrix(self, spmatrix):
        rating = spmatrix.toarray()
        self.rating_matrix = torch.tensor(rating)

    def build_histroy_items(self, dataset):
        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix()

    def get_rating_matrix(self, user):
        r"""Get a batch of user's feature with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The user's feature of a batch of user, shape: [batch_size, n_items]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = self.history_item_id[user.cpu()].flatten()
        row_indices = torch.arange(user.shape[0]).repeat_interleave(
            self.history_item_id.shape[1], dim=0
        )
        rating_matrix = torch.zeros(1).repeat(user.shape[0], self.total_num_items)
        rating_matrix.index_put_(
            (row_indices, col_indices), self.history_item_value[user.cpu()].flatten()
        )
        rating_matrix = rating_matrix.to(self.device)
        return rating_matrix


class MFGSLAE(CrossDomainRecommender, AutoEncoderMixin):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MFGSLAE, self).__init__(config, dataset)

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field
        self.source_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(np.float32)
        self.target_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(np.float32)
        self.convert_sparse_matrix_to_rating_matrix(self.source_interaction_matrix + self.target_interaction_matrix)
        
        # load parameters info
        self.device = config['device']
        self.lat_dim = config["latent_dimension"]
        self.drop_out = config["dropout_prob"]
        self.tau = config['tau']
        self.factor = config['factor']
        self.epsilon = config['epsilon']
        self.alpha = config['alpha']
        self.mask = config['mask']
        self.mask_type = config['mask_type']
        self.ablation_mask = config['ablation_mask']
        self.reg_mask = config['reg_mask']
        self.l1_rate = config['l1_rate']
        self.weight_decay = config['weight_decay']
        self.ratio = config['ratio']
        self.ratio_threshold = config['ratio_threshold']

        # define layers and loss
        self.hidden_dim = self.lat_dim
        self.source_encoder = nn.Linear(self.total_num_items, self.hidden_dim)
        self.target_encoder = nn.Linear(self.total_num_items, self.hidden_dim)
        self.gsl_encoder = nn.Linear(self.total_num_items, self.hidden_dim)
        self.act = nn.LeakyReLU()
        self.graph_layer = KernelAttention(
            self.hidden_dim,
            self.hidden_dim,
            self.factor,
            nb_random_features=64
        )
        self.subspace_projector = nn.ModuleList()
        for idx in range(self.factor):
            self.subspace_projector.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim // self.factor),
                    nn.ReLU(),
                )
            )
        self.norm = nn.ModuleList()
        self.norm.append(nn.LayerNorm(self.hidden_dim))
        self.norm.append(nn.LayerNorm(self.hidden_dim))
        self.source_aug_norm = nn.LayerNorm(self.hidden_dim)
        self.target_aug_norm = nn.LayerNorm(self.hidden_dim)
        self.proj_source = nn.Linear(self.lat_dim, self.lat_dim, bias=False)
        self.proj_target = nn.Linear(self.lat_dim, self.lat_dim, bias=False)
        self.mask_source = nn.Parameter(torch.empty(1, self.factor))
        self.mask_target = nn.Parameter(torch.empty(1, self.factor))
        self.decoder_source = nn.Linear(self.lat_dim, self.source_num_items)
        self.decoder_target = nn.Linear(self.lat_dim, self.target_num_items)
        self.binary = Binarize.apply

        # parameters initialization
        self.apply(xavier_normal_initialization)
        nn.init.normal_(self.mask_source, 0, 0.01)
        nn.init.normal_(self.mask_target, 0, 0.01)
        self.get_sparse_norm_rating_matrix()

    def compactness(self, emb):
        emb = F.normalize(emb, dim=-1)
        loss = 0
        emb_all = torch.flatten(emb, 0, 1) # [NK, D]
        NK, D = emb_all.shape
        loss -= 0.5 * torch.logdet(torch.eye(D, device=self.device) + (D / NK / self.epsilon * emb_all.T) @ emb_all)

        N, K, D = emb.shape
        emb = torch.einsum('abc->bca', emb) # [K, D, N]
        loss_factor = torch.einsum('abc,adc->abd', D / N / self.epsilon * emb, emb) # [K, D, D]
        loss_factor = 0.5 * torch.logdet(torch.eye(D, device=self.device).unsqueeze(0) + loss_factor)
        loss += loss_factor.sum()

        return loss

    def get_sparse_norm_rating_matrix(self):
        rating_matrix = F.normalize(self.rating_matrix).to_sparse()
        self.sparse_norm_rating_matrix = rating_matrix.to(self.device)
        rating_matrix = torch.tensor(self.source_interaction_matrix.toarray())
        self.sparse_norm_rating_source = F.normalize(rating_matrix).to_sparse().to(self.device)
        rating_matrix = torch.tensor(self.target_interaction_matrix.toarray())
        self.sparse_norm_rating_target = F.normalize(rating_matrix).to_sparse().to(self.device)

        self.rating_matrix_source = torch.cat([
            self.rating_matrix[:, :self.overlapped_num_items],
            self.rating_matrix[:, self.target_num_items:]
        ], dim=-1)
        self.rating_matrix_target = self.rating_matrix[:self.target_num_users][:, :self.target_num_items]

    def sprase_drop(self, A : torch.Tensor):
        if self.training:
            dist = Binomial(probs=1 - self.drop_out)
            mask = dist.sample(A.values().size()).bool()
            A_drop = torch.sparse_coo_tensor(A.indices()[:, mask], A.values()[mask] * 1.0 / (1 - self.drop_out), size=A.size())
            return A_drop.to(A.device)
        else:
            return A

    def factor_gsl(self, h):
        h = self.norm[0](h)
        self.h_factor = self.factor_division(h)
        h_enhanced = self.graph_layer(self.h_factor, self.tau)
        h = h + h_enhanced
        h = self.act(h)
        h = self.norm[1](h)
        return h

    def factor_division(self, h):
        h_factor_list = []
        for idx in range(self.factor):
            h_factor = self.subspace_projector[idx](h)
            h_factor_list.append(h_factor)
        h_factor = torch.stack(h_factor_list, dim=1)
        return h_factor

    def factor_selection(self, h):
        h_source = self.proj_source(h).reshape(h.shape[0], self.factor, -1)
        h_target = self.proj_target(h).reshape(h.shape[0], self.factor, -1)
        mask_source = self.binary(self.mask_source).unsqueeze(-1)
        mask_target = self.binary(self.mask_target).unsqueeze(-1)
        h_source_aug = self.source_aug_norm((h_source * mask_source).flatten(-2, -1))
        h_target_aug = self.target_aug_norm((h_target * mask_target).flatten(-2, -1))
        return h_source_aug, h_target_aug

    def forward(self, source_user=[], target_user=[]):
        # Source bootstrapping channel
        input = self.sprase_drop(self.sparse_norm_rating_source)
        h = torch.sparse.mm(input, self.source_encoder.weight.T) + self.source_encoder.bias
        h_source = self.act(h)

        # Target bootstrapping channel
        input = self.sprase_drop(self.sparse_norm_rating_target)
        h = torch.sparse.mm(input, self.target_encoder.weight.T) + self.target_encoder.bias
        h_target = self.act(h)

        # GSL augmentation channel
        input = self.sprase_drop(self.sparse_norm_rating_matrix)
        h = torch.sparse.mm(input, self.gsl_encoder.weight.T) + self.gsl_encoder.bias
        res = h[0:1] # User in position 1 is meaningless
        h_gsl = self.act(h[1:])
        h_gsl = self.factor_gsl(h_gsl)
        h_gsl = torch.cat([res, h_gsl])
        self.h_gsl = h_gsl.reshape(h_gsl.shape[0], self.factor, -1)
        if self.factor > 1:
            h_source_aug, h_target_aug = self.factor_selection(h_gsl)
        else:
            h_source_aug, h_target_aug = self.proj_source(h_gsl), self.proj_target(h_gsl)
            
        # Bootstrapping augmentation
        h_source = self.ratio * h_source + (1 - self.ratio) * h_source_aug
        h_target = self.ratio * h_target + (1 - self.ratio) * h_target_aug

        # Decoder
        reconstructed_source = h_source[source_user] @ self.decoder_source.weight.T + self.decoder_source.bias
        reconstructed_target = h_target[target_user] @ self.decoder_target.weight.T + self.decoder_target.bias

        # return reconstructed
        return reconstructed_source, reconstructed_target

    def process_source_user_id(self, id):
        id[id >= self.overlapped_num_users] += self.target_num_users - self.overlapped_num_users
        return id

    def epoch_start(self):
        if self.ratio > self.ratio_threshold:
            self.ratio = self.ratio * self.ratio
        else:
            self.ratio = self.ratio_threshold

    def calculate_loss(self, interaction):
        source_user_id = interaction[self.SOURCE_USER_ID]
        target_user_id = interaction[self.TARGET_USER_ID]
        # source_user_id = self.process_source_user_id(source_user_id)

        rating_matrix_source = self.rating_matrix_source[source_user_id.cpu()].to(self.device)
        rating_matrix_target = self.rating_matrix_target[target_user_id.cpu()].to(self.device)

        # Recommendation task
        reconstructed_source, reconstructed_target = self.forward(source_user_id, target_user_id)

        ce_loss_source = -(F.log_softmax(reconstructed_source, 1) * rating_matrix_source).sum(1).mean()
        ce_loss_target = -(F.log_softmax(reconstructed_target, 1) * rating_matrix_target).sum(1).mean()

        if self.factor > 1:
            compactness_loss = self.alpha * (self.compactness(self.h_factor) + self.compactness(self.h_gsl))
        else:
            compactness_loss = torch.tensor(0, device=self.device)
        l1_loss = self.l1_rate * (
            F.l1_loss(self.mask_source, torch.zeros_like(self.mask_source, device=self.device)) +
            F.l1_loss(self.mask_target, torch.zeros_like(self.mask_target, device=self.device))
        )
        
        return (0.1* ce_loss_source, 0.9 * ce_loss_target, 0.5 * compactness_loss, 0.2 * l1_loss)

    # def full_sort_predict_source(self, interaction):
    #     user = interaction[self.SOURCE_USER_ID]

    #     reconstructed_source, _ = self.forward(source_user=user)
    #     return reconstructed_source.reshape(-1)

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        
        _, reconstructed_target = self.forward(target_user=user)
        return reconstructed_target[item]

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        
        _, reconstructed_target = self.forward(target_user=user)
        return reconstructed_target.reshape(-1)
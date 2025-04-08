

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss, BPRLoss
from recbole.utils import InputType
from .attention import CDUserAttention, CDUserItemAttention
from .utils import uniformity, alignment

class VUG(CrossDomainRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(VUG, self).__init__(config, dataset)

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.device = config['device']

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.domain_lambda_source = config['lambda_source']  # float32 type: the weight of source embedding in transfer function
        self.domain_lambda_target = config['lambda_target']  # float32 type: the weight of target embedding in transfer function
        self.drop_rate = config['drop_rate']  # float32 type: the dropout rate
        self.connect_way = config['connect_way']  # str type: the connect way for all layers

        # define layers and loss
        self.source_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)
        self.target_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)

        self.source_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)
        self.target_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)

        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)

            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)

        self.dropout = nn.Dropout(p=self.drop_rate)
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.reg_loss = EmbLoss()

        # generate intermediate data
        self.source_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(np.float32)
        self.target_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(np.float32)
        self.source_norm_adj_matrix = self.get_norm_adj_mat(self.source_interaction_matrix, self.total_num_users,
                                                       self.total_num_items).to(self.device)
        self.target_norm_adj_matrix = self.get_norm_adj_mat(self.target_interaction_matrix, self.total_num_users,
                                                       self.total_num_items).to(self.device)

        self.source_user_degree_count = torch.from_numpy(self.source_interaction_matrix.sum(axis=1)).to(self.device)
        self.target_user_degree_count = torch.from_numpy(self.target_interaction_matrix.sum(axis=1)).to(self.device)
        self.source_item_degree_count = torch.from_numpy(self.source_interaction_matrix.sum(axis=0)).transpose(0, 1).to(self.device)
        self.target_item_degree_count = torch.from_numpy(self.target_interaction_matrix.sum(axis=0)).transpose(0, 1).to(self.device)

        # storage variables for full sort evaluation acceleration
        self.target_restore_user_e = None
        self.target_restore_item_e = None

        # 1) NEW: Use user–item dual-attention generator
        self.source_generator = CDUserItemAttention(self.latent_dim)

        # config for generator
        self.gen_weight = config["gen_weight"]
        self.enhance_mode = config["enhance_mode"]
        self.enhance_weight = config["enhance_weight"]
        self.user_weight_attn = config["user_weight_attn"]
        self.gen_loss = nn.MSELoss()



        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['target_restore_user_e', 'target_restore_item_e']

        self.phase = 'both'
        self.is_transfer = config['is_transfer']

        # 2) Build a boolean mask of shape [num_total_user, 1]
        #    where True indicates target-only users.
        self.target_nonoverlap_mask = torch.zeros(
            (self.target_user_embedding.weight.size(0), 1),
            dtype=torch.bool,
            device=self.device
        )
        self.target_nonoverlap_mask[self.num_overlap_user:  self.num_overlap_user + self.num_target_only_user] = True

        # 3) Broadcast self.target_nonoverlap_mask to match embedding dimension => [num_total_user, embedding_dim]
        self.target_nonoverlap_mask = self.target_nonoverlap_mask.expand(-1, self.target_user_embedding.weight.size(1)).to(self.target_user_embedding.weight.dtype)


    def get_norm_adj_mat(self, interaction_matrix, n_users=None, n_items=None):
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

    def get_ego_embeddings(self, domain='source'):
        if domain == 'source':
            user_embeddings = self.source_user_embedding.weight
            item_embeddings = self.source_item_embedding.weight
            norm_adj_matrix = self.source_norm_adj_matrix

            if self.enhance_mode == "asrealsource":
                # Generate counterfactual embeddings for non-overlapping target users
                _, gen_source_emb, _, _ = self.generate_source_embeddings()

                # Overwrite the *real* source embeddings for non-overlapping target users
                adjusted_user_embeddings = user_embeddings + self.enhance_weight * self.target_nonoverlap_mask * gen_source_emb
            else: 
                adjusted_user_embeddings = user_embeddings

        else:
            user_embeddings = self.target_user_embedding.weight
            item_embeddings = self.target_item_embedding.weight
            norm_adj_matrix = self.target_norm_adj_matrix

            _, gen_source_emb, _, _ = self.generate_source_embeddings()  
            if self.enhance_mode == "add":
                adjusted_user_embeddings = user_embeddings + self.enhance_weight * self.target_nonoverlap_mask * gen_source_emb
            else:
                adjusted_user_embeddings = user_embeddings

        ego_embeddings = torch.cat([adjusted_user_embeddings, item_embeddings], dim=0)
        return ego_embeddings, norm_adj_matrix

    def graph_layer(self, adj_matrix, all_embeddings):
        side_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
        new_embeddings = side_embeddings + torch.mul(all_embeddings, side_embeddings)
        new_embeddings = all_embeddings + new_embeddings
        new_embeddings = self.dropout(new_embeddings)
        return new_embeddings

    def transfer_layer(self, source_all_embeddings, target_all_embeddings):
        source_user_embeddings, source_item_embeddings = torch.split(source_all_embeddings, [self.total_num_users, self.total_num_items])
        target_user_embeddings, target_item_embeddings = torch.split(target_all_embeddings, [self.total_num_users, self.total_num_items])

        source_user_embeddings_lam = self.domain_lambda_source * source_user_embeddings + (1 - self.domain_lambda_source) * target_user_embeddings
        target_user_embeddings_lam = self.domain_lambda_target * target_user_embeddings + (1 - self.domain_lambda_target) * source_user_embeddings
        source_item_embeddings_lam = self.domain_lambda_source * source_item_embeddings + (1 - self.domain_lambda_source) * target_item_embeddings
        target_item_embeddings_lam = self.domain_lambda_target * target_item_embeddings + (1 - self.domain_lambda_target) * source_item_embeddings

        source_user_laplace = self.source_user_degree_count
        target_user_laplace = self.target_user_degree_count
        user_laplace = source_user_laplace + target_user_laplace + 1e-7
        source_user_embeddings_lap = (source_user_laplace * source_user_embeddings + target_user_laplace * target_user_embeddings) / user_laplace
        target_user_embeddings_lap = source_user_embeddings_lap
        source_item_laplace = self.source_item_degree_count
        target_item_laplace = self.target_item_degree_count
        item_laplace = source_item_laplace + target_item_laplace + 1e-7
        source_item_embeddings_lap = (source_item_laplace * source_item_embeddings + target_item_laplace * target_item_embeddings) / item_laplace
        target_item_embeddings_lap = source_item_embeddings_lap

        if self.enhance_mode == "asrealsource":
            source_specific_user_embeddings = source_user_embeddings[self.target_num_users:]
            target_specific_user_embeddings = target_user_embeddings[self.target_num_users:]
            source_overlap_user_embeddings = (source_user_embeddings_lam[:self.target_num_users] + source_user_embeddings_lap[:self.target_num_users]) / 2
            target_overlap_user_embeddings = (target_user_embeddings_lam[:self.target_num_users] + target_user_embeddings_lap[:self.target_num_users]) / 2
        else:
            source_specific_user_embeddings = source_user_embeddings[self.overlapped_num_users:]
            target_specific_user_embeddings = target_user_embeddings[self.overlapped_num_users:]
            source_overlap_user_embeddings = (source_user_embeddings_lam[:self.overlapped_num_users] + source_user_embeddings_lap[:self.overlapped_num_users]) / 2
            target_overlap_user_embeddings = (target_user_embeddings_lam[:self.overlapped_num_users] + target_user_embeddings_lap[:self.overlapped_num_users]) / 2

        source_specific_item_embeddings = source_item_embeddings[self.overlapped_num_items:]
        target_specific_item_embeddings = target_item_embeddings[self.overlapped_num_items:]
        source_overlap_item_embeddings = (source_item_embeddings_lam[:self.overlapped_num_items] + source_item_embeddings_lap[:self.overlapped_num_items]) / 2
        target_overlap_item_embeddings = (target_item_embeddings_lam[:self.overlapped_num_items] + target_item_embeddings_lap[:self.overlapped_num_items]) / 2
        
        source_transfer_user_embeddings = torch.cat([source_overlap_user_embeddings, source_specific_user_embeddings], dim=0)
        target_transfer_user_embeddings = torch.cat([target_overlap_user_embeddings, target_specific_user_embeddings], dim=0)
        source_transfer_item_embeddings = torch.cat([source_overlap_item_embeddings, source_specific_item_embeddings], dim=0)
        target_transfer_item_embeddings = torch.cat([target_overlap_item_embeddings, target_specific_item_embeddings], dim=0)

        source_alltransfer_embeddings = torch.cat([source_transfer_user_embeddings, source_transfer_item_embeddings], dim=0)
        target_alltransfer_embeddings = torch.cat([target_transfer_user_embeddings, target_transfer_item_embeddings], dim=0)
        return source_alltransfer_embeddings, target_alltransfer_embeddings

    def forward(self):
        source_all_embeddings, source_norm_adj_matrix = self.get_ego_embeddings(domain='source')
        target_all_embeddings, target_norm_adj_matrix = self.get_ego_embeddings(domain='target')

        source_embeddings_list = [source_all_embeddings]
        target_embeddings_list = [target_all_embeddings]
        for layer_idx in range(self.n_layers):
            source_all_embeddings = self.graph_layer(source_norm_adj_matrix, source_all_embeddings)
            target_all_embeddings = self.graph_layer(target_norm_adj_matrix, target_all_embeddings)

            # only transfer feature when phase is BOTH and is_transfer is True
            if self.phase == 'BOTH' and self.is_transfer:
                source_all_embeddings, target_all_embeddings = self.transfer_layer(source_all_embeddings, target_all_embeddings)

            source_norm_embeddings = nn.functional.normalize(source_all_embeddings, p=2, dim=1)
            target_norm_embeddings = nn.functional.normalize(target_all_embeddings, p=2, dim=1)
            source_embeddings_list.append(source_norm_embeddings)
            target_embeddings_list.append(target_norm_embeddings)

        if self.connect_way == 'concat':
            source_lightgcn_all_embeddings = torch.cat(source_embeddings_list, 1)
            target_lightgcn_all_embeddings = torch.cat(target_embeddings_list, 1)
        elif self.connect_way == 'mean':
            source_lightgcn_all_embeddings = torch.stack(source_embeddings_list, dim=1)
            source_lightgcn_all_embeddings = torch.mean(source_lightgcn_all_embeddings, dim=1)
            target_lightgcn_all_embeddings = torch.stack(target_embeddings_list, dim=1)
            target_lightgcn_all_embeddings = torch.mean(target_lightgcn_all_embeddings, dim=1)

        source_user_all_embeddings, source_item_all_embeddings = torch.split(source_lightgcn_all_embeddings,
                                                                   [self.total_num_users, self.total_num_items])
        target_user_all_embeddings, target_item_all_embeddings = torch.split(target_lightgcn_all_embeddings,
                                                                   [self.total_num_users, self.total_num_items])

        return source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings

    def calculate_loss(self, interaction):
        self.init_restore_e()
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings = self.forward()

        losses = []

        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        source_u_embeddings = source_user_all_embeddings[source_user]
        source_i_embeddings = source_item_all_embeddings[source_item]
        target_u_embeddings = target_user_all_embeddings[target_user]
        target_i_embeddings = target_item_all_embeddings[target_item]

        # calculate BCE Loss in source domain
        source_output = self.sigmoid(torch.mul(source_u_embeddings, source_i_embeddings).sum(dim=1))
        source_bce_loss = self.loss(source_output, source_label)

        # calculate Reg Loss in source domain
        u_ego_embeddings = self.source_user_embedding(source_user)
        i_ego_embeddings = self.source_item_embedding(source_item)
        source_reg_loss = self.reg_loss(u_ego_embeddings, i_ego_embeddings)

        source_loss = source_bce_loss + self.reg_weight * source_reg_loss
        losses.append(source_loss)

        # calculate BCE Loss in target domain
        target_output = self.sigmoid(torch.mul(target_u_embeddings, target_i_embeddings).sum(dim=1))
        target_bce_loss = self.loss(target_output, target_label)

        # calculate Reg Loss in target domain
        u_ego_embeddings = self.target_user_embedding(target_user)
        i_ego_embeddings = self.target_item_embedding(target_item)
        target_reg_loss = self.reg_loss(u_ego_embeddings, i_ego_embeddings)

        target_loss = target_bce_loss + self.reg_weight * target_reg_loss
        losses.append(target_loss)

        return tuple(losses)

    def predict(self, interaction):
        result = []
        _, _, target_user_all_embeddings, target_item_all_embeddings = self.forward()
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]

        u_embeddings = target_user_all_embeddings[user]
        i_embeddings = target_item_all_embeddings[item]

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]

        restore_user_e, restore_item_e = self.get_restore_e()
        u_embeddings = restore_user_e[user]
        i_embeddings = restore_item_e[:self.target_num_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)

    def init_restore_e(self):
        # clear the storage variable when training
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None

    def get_restore_e(self):
        if self.target_restore_user_e is None or self.target_restore_item_e is None:
            _, _, self.target_restore_user_e, self.target_restore_item_e = self.forward()
        return self.target_restore_user_e, self.target_restore_item_e

    def set_phase(self, phase):
        self.phase = phase


    def calculate_gen_loss(self, interaction):
        r"""
        For overlapping users in the batch, encourage the generated source embedding to match the real one (MSE).
        """
        user, generated_s_emb, overlap_indices, non_overlap_indices = self.generate_source_embeddings(interaction)

        # Build a mask for only overlapped users => real source embeddings
        if overlap_indices.shape[0] == 0:
            # If no overlapped user in this batch, attention loss = 0
            return torch.tensor(0.0, device=interaction[self.TARGET_USER_ID].device)

        # Real source embeddings for overlapped users in the batch
        real_source_emb_for_overlapped = self.source_user_embedding(user[overlap_indices])

        # Generated embeddings for overlapped subset
        gen_for_overlapped = generated_s_emb[overlap_indices]

        loss = self.gen_weight * self.gen_loss(gen_for_overlapped, real_source_emb_for_overlapped) + (1 - self.gen_weight) * uniformity(gen_for_overlapped)

        return loss
    

    def generate_source_embeddings(self, interaction=None):
        r"""
        Use the dual-attention module (CDUserItemAttention) to generate source embeddings
        for *all* users in the batch (both overlapped & non-overlapped).

        Steps:
        1) Gather user IDs => shape (B,)
        2) Gather target-domain user embeddings (Q_user)
        3) Compute item-level user embedding (Q_item) by averaging item embeddings for each user
        4) Identify overlapping users => build (K_user, K_item, V_user)
        5) Dual-attention => output "generated" source embeddings for each user => shape (B, d)
        """
        # 1) user IDs
        if interaction is None:
            # If no batch is given, generate for *all* users
            user = torch.arange(
                self.total_num_users,
                device=self.target_user_embedding.weight.device
            )
        else:
            user = interaction[self.TARGET_USER_ID]  # shape: (B,)

        # 2) Gather "user-level" target embeddings => Q_user
        user_t_emb = self.target_user_embedding(user)  # (B, d)

        # 3) Compute a simple "item-level" embedding for each user in the batch => Q_item
        #    by averaging the target_item_embedding of each item they interacted with.
        Q_item = self._compute_item_level_user_emb(user)

        # 4) Identify which are overlapped users in this batch
        overlapped_mask = (user < self.overlapped_num_users)
        overlap_indices = torch.nonzero(overlapped_mask).flatten()      # (M,)
        non_overlap_indices = torch.nonzero(~overlapped_mask).flatten() # (B-M,)

        # Overlapping users => their user-level K_user & item-level K_item
        K_user = user_t_emb[overlap_indices]                        # (M, d), target user embedding
        K_item = self._compute_item_level_user_emb(user[overlap_indices])
        V_user = self.source_user_embedding(user[overlap_indices])  # (M, d) real source embedding

        # If no overlapped users in batch, attention has no M to attend over => trivial zero
        if overlap_indices.shape[0] == 0:
            # entire batch is non-overlapping => no real source user to attend
            generated_s_emb = torch.zeros_like(user_t_emb)
            return user, generated_s_emb, overlap_indices, non_overlap_indices

        # 5) Dual-attention => shape (B, d)
        generated_s_emb, attn_weights = self.source_generator(
            Q_user=user_t_emb,    # user-level query
            K_user=K_user,        # user-level key
            V_user=V_user,        # user-level value (source domain)
            Q_item=Q_item,
            K_item=K_item,
            alpha=self.user_weight_attn
        )

        return user, generated_s_emb, overlap_indices, non_overlap_indices
    
    def _compute_item_level_user_emb(self, user_ids):
        """
        Uses the target_norm_adj_matrix to perform a 1-layer GCN aggregation
        and takes the user-part of the aggregated embeddings as the 'item-level' representation
        for each user. We then index only the user_ids in the batch.
        """

        # 1. Concatenate user & item embeddings in target domain
        user_ego = self.target_user_embedding.weight      # shape: (num_users, d)
        item_ego = self.target_item_embedding.weight      # shape: (num_items, d)
        target_ego_embeddings = torch.cat([user_ego, item_ego], dim=0)  # shape: (num_users + num_items, d)

        # 2. Aggregate via 1-layer GCN with self.target_norm_adj_matrix
        side_embeddings = torch.sparse.mm(self.target_norm_adj_matrix, target_ego_embeddings)
        # side_embeddings => shape: (num_users + num_items, d)

        # 3. Split back into user vs. item
        user_agg, item_agg = torch.split(
            side_embeddings, [self.total_num_users, self.total_num_items], dim=0
        )

        # 4. The user-part (user_agg) is a GCN-aggregated embedding that indirectly captures
        #    item-level signals. Now just pick the rows corresponding to user_ids in the current batch.
        out = user_agg[user_ids]  # shape: (batch_size, d)
        return out
    

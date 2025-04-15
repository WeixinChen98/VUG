
r"""
EMCDR
################################################
Reference:
    Tong Man et al. "Cross-Domain Recommendation: An Embedding and Mapping Approach" in IJCAI 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender




class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SingleHeadAttention, self).__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=True)
        self.scale = np.sqrt(embed_dim)

    def forward(self, Q, K, V):
        Q_ = self.W_q(Q)  # (B, d)
        K_ = self.W_k(K)  # (M, d)
        V_ = self.W_v(V)  # (M, d)
        attn_logits = torch.matmul(Q_, K_.transpose(0, 1)) / self.scale
        attn_weights = F.softmax(attn_logits, dim=1)
        out = torch.matmul(attn_weights, V_)  # (B, d)
        return out, attn_weights


class EMCDRA(CrossDomainRecommender):
    r"""EMCDR learns an mapping function from source latent space
        to target latent space.

    """

    def __init__(self, config, dataset):
        super(EMCDRA, self).__init__(config, dataset)

        assert self.overlapped_num_items == 1 or self.overlapped_num_users == 1, \
            "EMCDR model only support user overlapped or item overlapped dataset! "
        # we only consider mode in ['overlap_users']
        if self.overlapped_num_users > 1:
            self.mode = 'overlap_users'
        elif self.overlapped_num_items > 1:
            self.mode = 'overlap_items'
        else:
            self.mode = 'non_overlap'
        self.phase = 'both'

        # load parameters info
        self.latent_factor_model = config['latent_factor_model']
        if self.latent_factor_model == 'MF':
            input_type = InputType.POINTWISE
            # load dataset info
            self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
            self.TARGET_LABEL = dataset.target_domain_dataset.label_field
            self.loss = nn.MSELoss()
        else:
            input_type = InputType.PAIRWISE
            # load dataset info
            self.loss = BPRLoss()
        self.source_latent_dim = config['source_embedding_size']  # int type:the embedding size of source latent space
        self.target_latent_dim = config['target_embedding_size']  # int type:the embedding size of target latent space
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.map_func = config['mapping_function']
        if self.map_func == 'linear':
            self.mapping = nn.Linear(self.source_latent_dim, self.target_latent_dim, bias=False)
        else:
            assert config["mlp_hidden_size"] is not None
            mlp_layers_dim = [self.source_latent_dim] + config["mlp_hidden_size"] + [self.target_latent_dim]
            self.mapping = self.mlp_layers(mlp_layers_dim)

        # define layers and loss
        self.source_user_embedding = torch.nn.Embedding(self.total_num_users, self.source_latent_dim)
        self.source_item_embedding = torch.nn.Embedding(self.total_num_items, self.source_latent_dim)

        self.target_user_embedding = torch.nn.Embedding(self.total_num_users, self.target_latent_dim)
        self.target_item_embedding = torch.nn.Embedding(self.total_num_items, self.target_latent_dim)

        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)

            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)

        self.reg_loss = EmbLoss()
        self.map_loss = nn.MSELoss()

        # 7) NEW: single-head attention generator
        #    This "generator" will produce a "counterfactual" source embedding
        #    for each user in the batch (both overlapped & non-overlapped).
        self.attn_generator = SingleHeadAttention(self.target_latent_dim)
        #    Weight for attention-based generator's supervised loss
        # self.att_gen_weight = config.get("att_gen_weight", 1.0)
        self.att_gen_weight = 1.0


        # parameters initialization
        self.apply(xavier_normal_initialization)

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())

        return nn.Sequential(*mlp_modules)

    def set_phase(self, phase):
        self.phase = phase

        # # Freeze source-domain embeddings if in OVERLAP or BOTH
        # if phase in ['OVERLAP', 'BOTH']:
        #     for param in self.source_user_embedding.parameters():
        #         param.requires_grad = False
        #     for param in self.source_item_embedding.parameters():
        #         param.requires_grad = False
        # else:
        #     # Unfreeze source-domain embeddings
        #     for param in self.source_user_embedding.parameters():
        #         param.requires_grad = True
        #     for param in self.source_item_embedding.parameters():
        #         param.requires_grad = True

    def source_forward(self, user, item):
        user_e = self.source_user_embedding(user)
        item_e = self.source_item_embedding(item)

        return torch.mul(user_e, item_e).sum(dim=1)

    def target_forward(self, user, item):
        user_e = self.target_user_embedding(user)
        item_e = self.target_item_embedding(item)

        return torch.mul(user_e, item_e).sum(dim=1)


    def calculate_source_loss(self, interaction):
        if self.latent_factor_model == 'MF':
            source_user = interaction[self.SOURCE_USER_ID]
            source_item = interaction[self.SOURCE_ITEM_ID]
            source_label = interaction[self.SOURCE_LABEL]

            p_source = self.source_forward(source_user, source_item)

            loss_s = self.loss(p_source, source_label) + \
                     self.reg_weight * self.reg_loss(self.source_user_embedding(source_user),
                                                     self.source_item_embedding(source_item))
        else:
            source_user = interaction[self.SOURCE_USER_ID]
            source_pos_item = interaction[self.SOURCE_ITEM_ID]
            source_neg_item = interaction[self.SOURCE_NEG_ITEM_ID]

            pos_item_score = self.source_forward(source_user, source_pos_item)
            neg_item_score = self.source_forward(source_user, source_neg_item)
            loss_s = self.loss(pos_item_score, neg_item_score) + \
                     self.reg_weight * self.reg_loss(self.source_user_embedding(source_user),
                                                     self.source_item_embedding(source_pos_item))
        return loss_s

    def calculate_target_loss(self, interaction):
        if self.latent_factor_model == 'MF':
            target_user = interaction[self.TARGET_USER_ID]
            target_item = interaction[self.TARGET_ITEM_ID]
            target_label = interaction[self.TARGET_LABEL]

            p_target = self.target_forward(target_user, target_item)

            loss_t = self.loss(p_target, target_label) + \
                     self.reg_weight * self.reg_loss(self.target_user_embedding(target_user),
                                                     self.target_item_embedding(target_item))
        else:
            target_user = interaction[self.TARGET_USER_ID]
            target_pos_item = interaction[self.TARGET_ITEM_ID]
            target_neg_item = interaction[self.TARGET_NEG_ITEM_ID]

            pos_item_score = self.target_forward(target_user, target_pos_item)
            neg_item_score = self.target_forward(target_user, target_neg_item)
            loss_t = self.loss(pos_item_score, neg_item_score) + \
                     self.reg_weight * self.reg_loss(self.target_user_embedding(target_user),
                                                     self.target_item_embedding(target_pos_item))
        return loss_t

    def calculate_map_loss(self, interaction):
        """
        This function replaces the old version that used self.OVERLAP_ID.
        Instead, we distinguish overlapped vs. non-overlapped users by comparing
        user ID < self.overlapped_num_users.
        
        Overlapped users => same as original: MSE between mapped(source_user_e) and target_user_e.
        Non-overlapped users => MSE between generated_s_emb (via attention) and target_user_e.
        """

        # 1. Identify which users in this mini-batch are overlapped vs. non-overlapped
        target_user = interaction[self.TARGET_USER_ID]  # shape: (batch_size,)
        overlapped_mask = (target_user < self.overlapped_num_users)
        overlap_idx = torch.nonzero(overlapped_mask).flatten()      # indices in the mini-batch for overlapped
        # non_overlap_idx = torch.nonzero(~overlapped_mask).flatten() # indices in the mini-batch for non-overlapped

        # 2. For overlapped users, we do the original linear (or MLP) mapping loss: map(source_user) ~ target_user
        overlap_loss = torch.tensor(0.0, device=target_user.device)
        if overlap_idx.numel() > 0:
            source_user_e = self.source_user_embedding(target_user[overlap_idx])
            target_user_e = self.target_user_embedding(target_user[overlap_idx])
            map_e = self.mapping(source_user_e)
            overlap_loss = self.map_loss(map_e, target_user_e)

        return overlap_loss

    def generate_source_embeddings(self, interaction):
        r"""
        Use attention to generate source embeddings for *all* users in the batch (both overlapped and non-overlapped).

        Steps:
        1) Gather user IDs from batch => shape (B,) 
        2) Gather target-domain user embeddings (Q) => (B, d)
        3) Identify overlapped user IDs in the batch => gather their target-domain (K) and source-domain (V) embeddings
        4) Single-head dot-prod attention => for each user in Q, produce a "generated" source embedding => (B, d)
        """
        # 1) user IDs
        user = interaction[self.TARGET_USER_ID]  # shape: (B,)
        user_t_emb = self.target_user_embedding(user)  # Q => (B, d)

        # 2) Identify which are overlapped vs. not
        #    overlapped_mask => True if user < self.overlapped_num_users
        overlapped_mask = (user < self.overlapped_num_users)
        # gather overlapped user indices
        overlap_indices = torch.nonzero(overlapped_mask).flatten()  # shape: (M,)
        # gather non-overlapped user indices
        non_overlap_indices = torch.nonzero(~overlapped_mask).flatten()  # shape: (B-M,)

        # 3) Build K and V from *overlapping users only*
        #    We will use only the overlapping users (in *this batch*) as the keys & values
        k = user_t_emb[overlap_indices]  # target embeddings of overlapped => (M, d)
        v = self.source_user_embedding(user[overlap_indices])  # true source embeddings => (M, d)

        # 4) Dot-product attention => shape( B, d )
        generated_s_emb, attn_weights = self.attn_generator(user_t_emb.detach(), k.detach(), v.detach())
        # "generated_s_emb" is the "counterfactual" source embedding for each user in the batch

        return user, generated_s_emb, overlap_indices, non_overlap_indices

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

        # Old MSE-based loss:
        # loss = F.mse_loss(gen_for_overlapped, real_source_emb_for_overlapped)

        # MSE/cosine loss for overlapped users
        cos_sim = F.cosine_similarity(gen_for_overlapped, real_source_emb_for_overlapped.detach(), dim=-1)  # shape (M,)
        loss = 0.0 * (1 - cos_sim).mean()  # Minimize 1 - cos => maximize cos

        # for EMCDR only
        non_overlap_loss = torch.tensor(0.0, device=user.device)
        if non_overlap_indices.numel() > 0:
            t_user_e_nov = self.target_user_embedding(user[non_overlap_indices])
            gen_s_emb_nov = generated_s_emb[non_overlap_indices]  # shape: (#nonoverlap, d)
            # Use the same self.map_loss (likely MSE) to match generated source embeddings to the target user embeddings
            non_overlap_loss = self.map_loss(gen_s_emb_nov, t_user_e_nov)
        
        loss += 0.5 * non_overlap_loss

        # return loss, generated_s_emb
        return loss



    def calculate_loss(self, interaction):
        if self.phase == 'SOURCE':
            return self.calculate_source_loss(interaction)
        elif self.phase == 'OVERLAP' or self.phase == 'BOTH':
            map_loss = self.calculate_map_loss(interaction)
            return map_loss
        else:
            return self.calculate_target_loss(interaction)

    def predict(self, interaction):
        if self.phase == 'SOURCE':
            user = interaction[self.SOURCE_USER_ID]
            item = interaction[self.SOURCE_ITEM_ID]
            user_e = self.source_user_embedding(user)
            item_e = self.source_item_embedding(item)
            score = torch.mul(user_e, item_e).sum(dim=1)
        elif self.phase == 'TARGET':
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            user_e = self.target_user_embedding(user)
            item_e = self.target_item_embedding(item)
            score = torch.mul(user_e, item_e).sum(dim=1)
        else:
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]

            repeat_user = user.repeat(self.source_latent_dim, 1).transpose(0, 1)
            user_e = torch.where(repeat_user < self.overlapped_num_users, self.mapping(self.source_user_embedding(user)),
                                    self.target_user_embedding(user))
            item_e = self.target_item_embedding(item)

            score = torch.mul(user_e, item_e).sum(dim=1)

            # # 'OVERLAP' or 'both'
            # # We'll do the standard cross-domain scoring: user in source space x item in target space?
            # # Or per your original code, we handle as if target domain input => source + item.
            # # Here, let's replicate your "else" branch but incorporate the attention-based generator.
            # user = interaction[self.TARGET_USER_ID]
            # item = interaction[self.TARGET_ITEM_ID]

            # # 1) Generate or retrieve source embeddings
            # #    Overlapped => real source emb
            # #    Non-overlapped => generated emb
            # overlapped_mask = (user < self.overlapped_num_users)
            # user_t_emb = self.target_user_embedding(user)
            # # Build K, V from overlapped users in the *current mini-batch*
            # # But in "predict", we only have a single user or small batch, so let's do a smaller version:
            # # We do single-user or single-batch approach for demonstration:
            # user_list = user
            # # We'll gather only the overlapping ones in this batch:
            # overlap_indices = torch.nonzero(overlapped_mask).flatten()
            # k = user_t_emb[overlap_indices]  # target embedding (overlapped)
            # v = self.source_user_embedding(user_list[overlap_indices])  # real source embedding (overlapped)
            # # Then for the entire batch => generate
            # gen_s_emb, _ = self.attn_generator(user_t_emb, k, v)

            # # 2) For overlapped users, use real source emb; for non-overlapped, use gen emb
            # real_s_emb = self.source_user_embedding(user)
            # # shape (B, d)
            # final_s_emb = torch.where(
            #     overlapped_mask.unsqueeze(-1),  # shape (B,1)
            #     real_s_emb,                    # real if overlapped
            #     gen_s_emb                      # generated if non-overlapped
            # )

            # # 3) item embedding in target domain
            # item_e = self.target_item_embedding(item)
            # # 4) Dot product
            # score = torch.mul(self.mapping(final_s_emb), item_e).sum(dim=1)


        return score

    def full_sort_predict(self, interaction):
        if self.phase == 'SOURCE':
            user = interaction[self.SOURCE_USER_ID]
            user_e = self.source_user_embedding(user)
            overlap_item_e = self.source_item_embedding.weight[:self.overlapped_num_items]
            source_item_e = self.source_item_embedding.weight[self.target_num_items:]
            all_item_e = torch.cat([overlap_item_e, source_item_e], dim=0)
        elif self.phase == 'TARGET':
            user = interaction[self.TARGET_USER_ID]
            user_e = self.target_user_embedding(user)
            all_item_e = self.target_item_embedding.weight[:self.target_num_items]
        else:
            user = interaction[self.TARGET_USER_ID]
            repeat_user = user.repeat(self.source_latent_dim, 1).transpose(0, 1)
            user_e = torch.where(repeat_user < self.overlapped_num_users, self.mapping(self.source_user_embedding(user)),
                                    self.target_user_embedding(user))
            all_item_e = self.target_item_embedding.weight[:self.target_num_items]

            # # 'OVERLAP' or 'both'
            # user = interaction[self.TARGET_USER_ID]
            # overlapped_mask = (user < self.overlapped_num_users)
            # user_t_emb = self.target_user_embedding(user)

            # # Build K, V from overlapped in this batch (which is typically 1 user in recbole's full-sort scenario),
            # # so let's do single-user or a small batch approach again:
            # overlap_indices = torch.nonzero(overlapped_mask).flatten()
            # k = user_t_emb[overlap_indices]  # (M, d)
            # v = self.source_user_embedding(user[overlap_indices])  # (M, d)
            # gen_s_emb, _ = self.attn_generator(user_t_emb, k, v)

            # real_s_emb = self.source_user_embedding(user)
            # final_s_emb = torch.where(
            #     overlapped_mask.unsqueeze(-1),
            #     real_s_emb,
            #     gen_s_emb
            # )
            # # Now multiply with all items in target domain
            # all_item_e = self.target_item_embedding.weight[:self.target_num_items]
            # user_e = self.mapping(final_s_emb)


        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

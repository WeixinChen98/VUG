import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CDUserAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CDUserAttention, self).__init__()
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
    

class CDUserItemAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CDUserItemAttention, self).__init__()
        
        # User-level Q, K, V (original attention)
        self.W_q_user = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_k_user = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_v_user = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Item-level Q, K (second attention)
        self.W_q_item = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_k_item = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # We still use the same V from user side (above) for both attentions.
        
        self.scale = np.sqrt(embed_dim)

    def forward(
        self,
        Q_user,    # (batch_size, embed_dim) - user-level target embeddings for non-overlapping users
        K_user,    # (num_overlapping, embed_dim) - user-level target embeddings for overlapping users
        V_user,    # (num_overlapping, embed_dim) - source-domain embeddings for overlapping users
        Q_item,    # (batch_size, embed_dim) - item-level user embeddings (non-overlapping)
        K_item,    # (num_overlapping, embed_dim) - item-level user embeddings (overlapping)
        alpha=0.9
    ):
        """
        Q_user, K_user, V_user: original user-level embeddings (target->target->source).
        Q_item, K_item: item-level user embeddings (target side).
        
        alpha: interpolation weight between user-level attention weights and item-level attention weights.
        """
        
        # 1. User-Level Attention
        q_user = self.W_q_user(Q_user)  # (B, d)
        k_user = self.W_k_user(K_user)  # (M, d)
        v_user = self.W_v_user(V_user)  # (M, d)
        
        logits_user = torch.matmul(q_user, k_user.transpose(0, 1)) / self.scale  # (B, M)
        attn_weights_user = F.softmax(logits_user, dim=1)                       # (B, M)

        # 2. Item-Level Attention
        q_item = self.W_q_item(Q_item)  # (B, d)
        k_item = self.W_k_item(K_item)  # (M, d)
        
        logits_item = torch.matmul(q_item, k_item.transpose(0, 1)) / self.scale  # (B, M)
        attn_weights_item = F.softmax(logits_item, dim=1)                        # (B, M)

        # 3. Interpolate the two sets of attention weights
        #    attn_weights = alpha * (user-level) + (1 - alpha) * (item-level)
        attn_weights = alpha * attn_weights_user + (1.0 - alpha) * attn_weights_item  # (B, M)

        # 4. Use the final attention weights to aggregate V_user
        out = torch.matmul(attn_weights, v_user)  # (B, d)

        return out, attn_weights
"""
Gaussian Distribution Generator for SC-VUG
==========================================

This module implements the Gaussian distribution modeling components for SC-VUG:
1. DualViewRetrieval: Cosine similarity-based dual-view attention (NEW)
2. GatedAggregation: Learns adaptive fusion weights for user-based and item-based views
3. VarianceEstimator: Estimates uncertainty based on global sparsity and local inconsistency
4. GaussianGenerator: Complete pipeline for generating embeddings from N(μ, σ²)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualViewRetrieval(nn.Module):
    """
    Dual-View Retrieval using Learned Attention (Paper Implementation)
    ===================================================================
    
    This implements the paper's Dual-View Mean Estimation using LEARNED ATTENTION:
    
    Paper Formulation:
        v^user = Σ_{k ∈ U^o} α^user_{u,k} * e^S_k    (user-user attention)
        v^item = Σ_{k ∈ U^o} α^item_{u,k} * e^S_k    (item-item attention)
        
        where α^user, α^item are computed via:
            α = softmax(Q @ K^T / sqrt(d))  (scaled dot-product attention)
            Q, K, V are learned linear projections
    
    Key Features:
    - Learned Q/K/V projections for both user-level and item-level attention
    - Scaled dot-product attention (standard Transformer-style)
    - Returns both views independently for downstream gated aggregation
    
    Similar to CDUserItemAttention but returns both views separately
    (not interpolated) for Gated Aggregation to handle.
    
    Inputs:
        Q_user: User-level query embeddings (B, d)
        Q_item: Item-level query embeddings (B, d)
        K_user: User-level key embeddings from overlapping users (M, d)
        K_item: Item-level key embeddings from overlapping users (M, d)
        V: Value embeddings (source domain) from overlapping users (M, d)
    
    Outputs:
        v_user: User-based view (B, d)
        v_item: Item-based view (B, d)
        alpha_user: User-based attention weights (B, M)
        alpha_item: Item-based attention weights (B, M)
    """
    
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim (int): Embedding dimension
        """
        super(DualViewRetrieval, self).__init__()
        
        # User-level Q, K, V projections
        self.W_q_user = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_k_user = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_v_user = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Item-level Q, K projections (use same V as user-level)
        self.W_q_item = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_k_item = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Scale factor for scaled dot-product attention
        self.scale = np.sqrt(embed_dim)
    
    def forward(self, Q_user, Q_item, K_user, K_item, V):
        """
        Compute dual-view retrieval using learned attention.
        
        Args:
            Q_user (Tensor): User-level query embeddings, shape (B, d)
            Q_item (Tensor): Item-level query embeddings, shape (B, d)
            K_user (Tensor): User-level key embeddings (overlapping users), shape (M, d)
            K_item (Tensor): Item-level key embeddings (overlapping users), shape (M, d)
            V (Tensor): Value embeddings (source domain overlapping users), shape (M, d)
        
        Returns:
            v_user (Tensor): User-based view, shape (B, d)
            v_item (Tensor): Item-based view, shape (B, d)
            alpha_user (Tensor): User-based attention weights, shape (B, M)
            alpha_item (Tensor): Item-based attention weights, shape (B, M)
        """
        # 1. User-based view: scaled dot-product attention
        # Paper: v^user = Σ α^user_{u,k} * e^S_k
        
        q_user = self.W_q_user(Q_user)  # (B, d)
        k_user = self.W_k_user(K_user)  # (M, d)
        v_user_proj = self.W_v_user(V)  # (M, d)
        
        # Compute attention scores: Q @ K^T / sqrt(d)
        logits_user = torch.matmul(q_user, k_user.transpose(0, 1)) / self.scale  # (B, M)
        alpha_user = F.softmax(logits_user, dim=-1)  # (B, M)
        
        # Weighted aggregation
        v_user = torch.matmul(alpha_user, v_user_proj)  # (B, d)
        
        # 2. Item-based view: scaled dot-product attention
        # Paper: v^item = Σ α^item_{u,k} * e^S_k
        
        q_item = self.W_q_item(Q_item)  # (B, d)
        k_item = self.W_k_item(K_item)  # (M, d)
        # Use same V projection as user-level
        
        logits_item = torch.matmul(q_item, k_item.transpose(0, 1)) / self.scale  # (B, M)
        alpha_item = F.softmax(logits_item, dim=-1)  # (B, M)
        
        # Weighted aggregation (use same v_user_proj)
        v_item = torch.matmul(alpha_item, v_user_proj)  # (B, d)
        
        return v_user, v_item, alpha_user, alpha_item


class GatedAggregation(nn.Module):
    """
    Gated Aggregation Module
    
    Instead of using a fixed scalar γ for interpolation, this module learns
    a gate vector z ∈ [0,1]^d to dynamically fuse user-based and item-based views:
    
    μ = z ⊙ v^user + (1-z) ⊙ v^item
    
    where z = Sigmoid(MLP([v^user, v^item]))
    """
    
    def __init__(self, embed_dim, hidden_dim=None):
        """
        Args:
            embed_dim (int): Embedding dimension
            hidden_dim (int, optional): Hidden layer dimension. Defaults to embed_dim.
        """
        super(GatedAggregation, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = embed_dim
        
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, v_user, v_item):
        """
        Args:
            v_user (Tensor): User-based view, shape (B, d)
            v_item (Tensor): Item-based view, shape (B, d)
        
        Returns:
            mu (Tensor): Gated fusion result, shape (B, d)
            gate (Tensor): Gate weights, shape (B, d)
        """
        # Concatenate two views
        concat_views = torch.cat([v_user, v_item], dim=-1)  # (B, 2d)
        
        # Learn gate weights
        gate = self.gate_network(concat_views)  # (B, d), values in [0, 1]
        
        # Gated fusion: μ = z ⊙ v^user + (1-z) ⊙ v^item
        mu = gate * v_user + (1 - gate) * v_item  # (B, d)
        
        return mu, gate


class VarianceEstimator(nn.Module):
    """
    Variance Estimation Module (Paper Implementation)
    
    Paper Formula:
        σ_u = Softplus(W_σ (φ(η) ⊕ (v^user - v^item)²) + b_σ) + ε
    
    Where:
    - φ(η): learnable embedding of overlap ratio η
    - (v^user - v^item)²: local disagreement (view inconsistency)
    - W_σ, b_σ: single linear layer (not 2-layer MLP)
    - Softplus: ensures positivity (Softplus(x) = log(1 + exp(x)))
    - ε: small constant for numerical stability
    
    Returns σ directly as in the paper (not log_var).
    """
    
    def __init__(self, embed_dim, overlap_ratio, hidden_dim=None):
        """
        Args:
            embed_dim (int): Embedding dimension
            overlap_ratio (float): Global overlap ratio η = |U_overlap| / |U_target|
            hidden_dim (int, optional): Not used (kept for backward compatibility)
        """
        super(VarianceEstimator, self).__init__()
        
        self.embed_dim = embed_dim
        self.overlap_ratio = overlap_ratio
        
        # Learnable embedding for overlap ratio φ(η)
        self.overlap_ratio_embedding = nn.Parameter(torch.randn(embed_dim))
        
        # Single linear layer: W_σ [φ(η) ⊕ (v^user - v^item)²] + b_σ
        self.W_sigma = nn.Linear(embed_dim * 2, embed_dim, bias=True)
        
        self.eps = 1e-6  # ε in paper formula
    
    def forward(self, v_user, v_item, batch_size):
        """
        Paper Formula:
            σ = Softplus(W_σ (φ(η) ⊕ (v^user - v^item)²) + b_σ) + ε
        
        Args:
            v_user (Tensor): User-based view, shape (B, d)
            v_item (Tensor): Item-based view, shape (B, d)
            batch_size (int): Batch size B
        
        Returns:
            sigma (Tensor): Standard deviation σ, shape (B, d)
        """
        # 1. Expand overlap ratio embedding φ(η) to batch size
        overlap_embedding = self.overlap_ratio_embedding.unsqueeze(0).expand(batch_size, -1)  # (B, d)
        
        # 2. Compute local inconsistency: (v^user - v^item)²
        view_diff_squared = (v_user - v_item).pow(2)  # (B, d)
        
        # 3. Concatenate: φ(η) ⊕ (v^user - v^item)²
        variance_input = torch.cat([overlap_embedding, view_diff_squared], dim=-1)  # (B, 2d)
        
        # 4. Single linear layer: W_σ [...] + b_σ
        raw_output = self.W_sigma(variance_input)  # (B, d)
        
        # 5. Apply Softplus + ε to get σ (paper formula)
        # Softplus(x) = log(1 + exp(x)), ensures positivity
        sigma = F.softplus(raw_output) + self.eps  # (B, d)
        
        return sigma


class GaussianGenerator(nn.Module):
    """
    Complete Gaussian Distribution Generator
    
    This module implements the full pipeline:
    1. Generate two views (v^user, v^item) via dual-attention
    2. Estimate mean μ via gated aggregation
    3. Estimate variance σ² via variance estimator
    4. Sample embedding via reparameterization trick: e = μ + σ ⊙ ε, ε ~ N(0,1)
    """
    
    def __init__(self, embed_dim, overlap_ratio, hidden_dim=None):
        """
        Args:
            embed_dim (int): Embedding dimension
            overlap_ratio (float): Global overlap ratio η
            hidden_dim (int, optional): Hidden dimension for gate and variance networks
        """
        super(GaussianGenerator, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Gated aggregation for mean estimation
        self.gated_aggregation = GatedAggregation(embed_dim, hidden_dim)
        
        # Variance estimation
        self.variance_estimator = VarianceEstimator(embed_dim, overlap_ratio, hidden_dim)
    
    def forward(self, v_user, v_item, return_distribution=False, training=True):
        """
        Paper Formula:
            μ = z ⊙ v^user + (1-z) ⊙ v^item
            σ = Softplus(W_σ [...] + b_σ) + ε
            e = μ + σ ⊙ ε, ε ~ N(0, I)
        
        Args:
            v_user (Tensor): User-based view, shape (B, d)
            v_item (Tensor): Item-based view, shape (B, d)
            return_distribution (bool): Whether to return (mu, sigma)
            training (bool): Whether in training mode (sample) or inference mode (use mean)
        
        Returns:
            generated_emb (Tensor): Sampled embedding, shape (B, d)
            mu (Tensor, optional): Mean μ, shape (B, d)
            sigma (Tensor, optional): Standard deviation σ, shape (B, d)
            gate (Tensor, optional): Gate weights z, shape (B, d)
        """
        batch_size = v_user.size(0)
        
        # 1. Estimate mean μ via gated aggregation
        mu, gate = self.gated_aggregation(v_user, v_item)  # (B, d)
        
        # 2. Estimate standard deviation σ (paper formula)
        sigma = self.variance_estimator(v_user, v_item, batch_size)  # (B, d)
        
        # 3. Reparameterization trick: e = μ + σ ⊙ ε, ε ~ N(0, I)
        if training:
            # Sample epsilon from standard normal: ε ~ N(0, I)
            epsilon = torch.randn_like(mu)  # (B, d)
            
            # Paper formula: e = μ + σ ⊙ ε
            generated_emb = mu + sigma * epsilon
        else:
            # During inference, use mean (no sampling for deterministic prediction)
            generated_emb = mu
        
        if return_distribution:
            return generated_emb, mu, sigma, gate
        else:
            return generated_emb


def compute_kl_divergence(mu, sigma):
    """
    Compute KL divergence between N(μ, σ²) and N(0, I)
    
    Paper Formula:
        KL(N(μ, σ²) || N(0, I)) = 0.5 * Σ(μ² + σ² - log(σ²) - 1)
                                  = 0.5 * Σ(μ² + σ² - 2*log(σ) - 1)
    
    Args:
        mu (Tensor): Mean μ, shape (B, d)
        sigma (Tensor): Standard deviation σ, shape (B, d)
    
    Returns:
        kl_loss (Tensor): Average KL divergence per sample
    """
    # Compute σ² and log(σ²)
    var = sigma.pow(2)              # σ²
    log_var = 2.0 * torch.log(sigma)  # log(σ²) = 2*log(σ)
    
    # KL divergence per dimension, summed over embedding dimension
    # KL = 0.5 * Σ(μ² + σ² - log(σ²) - 1)
    kl_per_sample = 0.5 * torch.sum(mu.pow(2) + var - log_var - 1, dim=-1)  # (B,)
    
    # Average over batch
    kl_loss = torch.mean(kl_per_sample)
    
    return kl_loss


def reparameterize(mu, log_var):
    """
    Reparameterization trick: sample from N(μ, σ²)
    
    z = μ + σ ⊙ ε, where ε ~ N(0, 1)
    
    Args:
        mu (Tensor): Mean, shape (*, d)
        log_var (Tensor): Log variance, shape (*, d)
    
    Returns:
        z (Tensor): Sampled latent variable, shape (*, d)
    """
    std = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(std)
    z = mu + std * epsilon
    return z



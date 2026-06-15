"""
action_heads.py

Implementations of various action heads, which serve as alternatives to VLM sequential token prediction.
"""

import math
import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS



def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations



class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512,
        use_pro_version=False,
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model = MLPResNet(
            num_blocks=24, 
            input_dim=input_dim*ACTION_DIM, 
            hidden_dim=hidden_dim, 
            output_dim=action_dim,
            use_pro_version=use_pro_version
            )

    def predict_action(
            self, 
            actions_hidden_states, 
            proprio=None, 
            proprio_projector=None,
            phase="Inference",
            use_moe =False
            ):
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]

        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
            device=device, dtype=actions_hidden_states.dtype
        ).detach()

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )  # (batch, chunk_len, action_dim * hidden_dim)

        if phase == "Training":
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype) 
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations) # (1, seq_len, dim)

        action = self.model(
            rearranged_actions_hidden_states,
            h_a=actions_hidden_states,
            p=proprio_features,
            h_t=task_hidden_states,
            use_moe = use_moe
            )

        return action
    

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(
            self, 
            num_blocks, 
            input_dim, 
            hidden_dim, 
            output_dim,
            use_pro_version=False
            ):
        
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            if use_pro_version:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
                
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, h_a=None, h_t=None, p= None,use_moe =False):
 
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        if use_moe:
            return x
        else:
            x = self.fc2(x)  # shape: (batch_size, output_dim)
            return x



def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)


    def rotate_half(x):
        # Swap even and odd dimensions and flip the signs
        x1 = x[..., ::2]   # Even subdimension
        x2 = x[..., 1::2]  # odd subdimension

        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)



class MLPResNetBlock(nn.Module):
    """
    One residual MLP block with cross-attention conditioning.

    This block applies multi-head attention over:
      - token features (self-attention),
      - task-related hidden states (h_t),
      - action/proprioception-related hidden states (h_a, p).
    The outputs are combined via a gating mechanism, projected back to the
    hidden dimension, and passed through a small feedforward sub-network with
    residual connection.

    Args:
        dim (int): Dimensionality of the hidden features. Must be divisible by num_heads.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        h_t (torch.Tensor, optional): Task-related hidden states of shape
                                      (batch_size, K, hidden_dim).
        h_a (torch.Tensor, optional): Action-related hidden states of shape
                                      (batch_size, 1, hidden_dim).
        p (torch.Tensor, optional): Additional conditioning features
                                    (e.g., proprioception), shape (batch_size, 1, hidden_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))



    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """

        g = self.gating_factor
        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        h = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h

        adapter_k = h_t
        adapter_v = h_t

        q_1 = self.q_proj(x) # (B, T, C)
        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_task = self.k_proj(task_k)    # (B, K, C)
        v_task = self.v_proj(task_v)    # (B, K, C)

        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)

        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        attn_scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * 1 # (B, H, T, K)
        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g # (B, H, T, K)

        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, attn_scores_adapter], dim=-1) # (B, H, T, T+K)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)

        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2) # (B, H, T+K, head_dim)
        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x) 

        return x



class MLPResNetBlock_Pro(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now with FiLM modulation."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Self-Attention: K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # ---- FiLM ----
        # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now.
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim * 2),  # output γ and β
            )


    def apply_film(self, x, gamma, beta):
        """FiLM: per-channel modulation"""
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


    def forward(self, x, h_a=None, h_t=None, p=None):
        """
        h_a: adapter tokens
        h_t: task tokens
        p:   possible conditioning vector (for FiLM)
        """
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        # concat h_a and p
        h_adapter = torch.cat((h_a, p),dim=1)

        h_task = h_t
        B, T, C = x.shape
        K_a = h_adapter.size(1) if h_a is not None else 0
        K_t = h_task.size(1) if h_task is not None else 0

        # Q
        q_1 = self.q_proj(x)

        # self tokens
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)

        # adapter tokens
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)

        # task tokens
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)


        # reshape -> multi-head
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)


        q_1 = reshape_heads(q_1, B, T)
        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)     
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        # attention scores
        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
        attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # combine V
        v_list = [v_tokens,v_adapter,v_task]
        v_combined = torch.cat(v_list, dim=2)

        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # # ---- FiLM ---- 
        # gamma_beta = self.film_gen(p)  # [B, 2C]
        # gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]
        # output = self.apply_film(output, gamma, beta)

        # residual + FFN
        x = self.ffn(output + x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossExpertAttention(nn.Module):
    """
    双向交叉注意力模块，融入门控权重进行特征融合。
    输入:
        main_features: [B, seqlen, D]
        aux_features: [B, N, seqlen, D]  (N是副专家数量)
        gate_weights: [B, N+1] (可选) 主专家(索引0)和副专家的门控权重，用于引导注意力。
    输出:
        fused_features: [B, seqlen, D]
    """

    def __init__(self, dim, num_heads=8,top_k=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.top_k = top_k
        # --- 双向注意力所需的投影层 ---
        # 主专家 -> 副专家
        self.q_proj_main = nn.Linear(dim, dim)
        self.k_proj_aux = nn.Linear(dim, dim)
        self.v_proj_aux = nn.Linear(dim, dim)

        # 副专家 -> 主专家
        self.q_proj_aux = nn.Linear(dim, dim)
        self.k_proj_main = nn.Linear(dim, dim)
        self.v_proj_main = nn.Linear(dim, dim)

        # 注意力输出投影
        self.out_proj_main = nn.Linear(dim, dim)
        self.out_proj_aux = nn.Linear(dim, dim)

        # LayerNorm
        self.norm_main = nn.LayerNorm(dim)
        self.norm_aux = nn.LayerNorm(dim)

        # --- 特征融合层 ---
        # 1. 特征融合：将主专家特征(D)和所有副专家特征(N*D)拼接后融合
        self.feature_fusion = nn.Linear(dim * (self.top_k), dim)  # 1个主专家 + 2个副专家

    def forward(self, main_features, aux_features, gate_weights=None):
        #import pdb;pdb.set_trace()
        B, seqlen, D = main_features.shape
        N = aux_features.shape[1]  # 副专家数量 (应为2)
        if gate_weights is not None:
            gate_weights = gate_weights.to(dtype=main_features.dtype, device=main_features.device)
        # -------------------------- 第一步：双向交叉注意力交互 --------------------------
        # 1. 主专家关注副专家 (Main -> Aux)
        q_main = self.q_proj_main(main_features).view(B, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        # 处理副专家特征
        aux_features_flat = aux_features.reshape(B, N * seqlen, D)
        k_aux = self.k_proj_aux(aux_features_flat).view(B, N * seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v_aux = self.v_proj_aux(aux_features_flat).view(B, N * seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        attn_main2aux = torch.matmul(q_main, k_aux.transpose(-2, -1)) * self.scale

        # --- 优化点 2：融入门控权重作为先验 ---
        if gate_weights is not None:
            # gate_weights: [B, N+1], 我们取副专家的权重 [B, N]
            aux_gate_weights = gate_weights[:, 1:]  # [B, N]
            # 将权重广播到 [B, 1, 1, N] 以匹配注意力分数的维度 [B, H, seqlen, N*seqlen]
            # 我们希望为每个副专家的所有时间步施加相同的权重
            aux_gate_weights = aux_gate_weights.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            # 重复权重以覆盖每个副专家内部的时间步
            aux_gate_weights = aux_gate_weights.repeat(1, self.num_heads, seqlen, 1)  # [B, H, seqlen, N]
            aux_gate_weights = aux_gate_weights.repeat_interleave(seqlen, dim=-1)  # [B, H, seqlen, N*seqlen]

            attn_main2aux = attn_main2aux * aux_gate_weights

        attn_main2aux = torch.softmax(attn_main2aux, dim=-1)
        out_main2aux = torch.matmul(attn_main2aux, v_aux)
        out_main2aux = out_main2aux.transpose(1, 2).contiguous().view(B, seqlen, D)
        out_main2aux = self.out_proj_main(out_main2aux)

        # 残差连接和归一化
        fused_main = self.norm_main(main_features + out_main2aux)

        # 2. 副专家关注主专家 (Aux -> Main)
        q_aux = self.q_proj_aux(aux_features_flat).view(B, N * seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k_main = self.k_proj_main(main_features).view(B, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v_main = self.v_proj_main(main_features).view(B, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        attn_aux2main = torch.matmul(q_aux, k_main.transpose(-2, -1)) * self.scale
        attn_aux2main = torch.softmax(attn_aux2main, dim=-1)
        out_aux2main = torch.matmul(attn_aux2main, v_main)
        out_aux2main = out_aux2main.transpose(1, 2).contiguous().view(B, N, seqlen, D)
        out_aux2main = self.out_proj_aux(out_aux2main)

        # 残差连接和归一化
        fused_aux = self.norm_aux(aux_features + out_aux2main)

        # -------------------------- 第二步：特征融合 --------------------------
        # 将副专家特征从 [B, N, seqlen, D] 展平为 [B, seqlen, N*D]
        aux_flat = fused_aux.transpose(1, 2).reshape(B, seqlen, -1)

        # 拼接主专家和所有副专家的特征 [B, seqlen, D + N*D]
        combined_features = torch.cat([fused_main, aux_flat], dim=-1)

        # 通过一个线性层进行融合
        fused_features = F.gelu(self.feature_fusion(combined_features))

        return fused_features


class LoadBalanceLoss(nn.Module):
    """负载均衡损失，鼓励专家被均匀使用"""

    def __init__(self, num_experts, alpha=0.05):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha

    def forward(self, gate_probs, expert_indices=None):
        """
        Args:
            gate_probs: (B, num_experts) - 每个样本对专家的概率分布
            expert_indices: (B, K) - 被选中的专家索引

        Returns:
            load_balance_loss: 标量
            metrics: 字典，包含统计信息
        """

        # 负载损失：确保每个专家被选中的次数相近
        if expert_indices is not None:
            load = torch.zeros(self.num_experts, device=gate_probs.device)
            load.scatter_add_(0, expert_indices.flatten(),
                              torch.ones_like(expert_indices.flatten(), dtype=torch.float))
            load_loss = torch.var(load) / (load.mean() ** 2 + 1e-10)
        else:
            load_loss = torch.tensor(0.0, device=gate_probs.device)

        total_loss = self.alpha * load_loss

        metrics = {
            'load_loss': load_loss.item(),
        }

        return total_loss, metrics


# class StageMoEActionHead(nn.Module):
#     """基于阶段的MoE动作头 - 使用外部action_probs"""

#     def __init__(
#             self,
#             input_dim=4096,
#             hidden_dim=4096,
#             action_dim=7,
#             use_pro_version=False,
#             stage_definitions=None,
#             num_action_ids=18,
#             top_k=2,
#             load_balance_alpha=0.01,
#     ):
#         super().__init__()
#         self.action_dim = action_dim
#         self.hidden_dim = hidden_dim
#         self.num_action_ids = num_action_ids
#         self.top_k = top_k

#         # 1. 定义阶段
#         if stage_definitions is not None:
#             self.stages = stage_definitions
#         else:
#             self.stages = {
#                 0: list(range(0, 6)),  # 预接触阶段
#                 1: list(range(6, 12)),  # 交互阶段
#                 2: list(range(12, 18))  # 后调整阶段
#             }
#         self.num_experts = len(self.stages)

#         # 2. 创建阶段专家
#         self.stage_experts = nn.ModuleList([
#             L1RegressionActionHead(
#                 input_dim=input_dim,
#                 hidden_dim=hidden_dim,
#                 action_dim=action_dim,
#                 use_pro_version=use_pro_version
#             ) for _ in range(self.num_experts)
#         ])

#         # 3. 负载均衡损失
#         self.load_balance_loss_fn = LoadBalanceLoss(
#             num_experts=self.num_experts,
#             alpha=load_balance_alpha
#         )

#         # 4. 跨专家注意力模块
#         self.cross_attn = CrossExpertAttention(
#             hidden_dim,
#             num_heads=8,
#             top_k =self.top_k
#         )

#         # 5. 动作预测头
#         self.action_projection = nn.Linear(hidden_dim, action_dim)

#         # 用于保存辅助损失
#         self.aux_loss = 0.0
#         self.load_balance_metrics = {}

#     def _aggregate_action_probs_to_stage_probs(self, action_probs):
#         """
#         将动作概率聚合到阶段概率

#         Args:
#             action_probs: (B, num_action_ids) - 每个动作ID的概率

#         Returns:
#             stage_probs: (B, num_experts) - 每个阶段/专家的概率
#         """
#         batch_size = action_probs.shape[0]
#         device = action_probs.device

#         stage_probs = torch.zeros(batch_size, self.num_experts, device=device)

#         # 将每个动作的概率累加到对应的阶段
#         for stage_idx, action_ids in self.stages.items():
#             # action_ids是该阶段包含的动作ID列表
#             for action_id in action_ids:
#                 if action_id < action_probs.shape[1]:
#                     stage_probs[:, stage_idx] += action_probs[:, action_id]

#         # 归一化
#         stage_probs = stage_probs / (stage_probs.sum(dim=-1, keepdim=True) + 1e-10)

#         return stage_probs

#     def _sparse_expert_execution(
#             self,
#             actions_hidden_states,  # (B, T, N, D)
#             top_k_indices,
#             proprio=None,
#             proprio_projector=None,
#             phase="Inference"
#     ):
#         """
#         稀疏执行Top-K专家

#         Args:
#             actions_hidden_states: (B, T, N, D) 输入特征
#             top_k_indices: (B, K) 每个样本选中的Top-K专家索引

#         Returns:
#             expert_outputs_tensor: (B, K, seq_len, D) 专家输出
#         """
#         batch_size = actions_hidden_states.shape[0]
#         expert_outputs = []

#         for b in range(batch_size):
#             # 提取第b个样本，保持维度结构
#             sample_hidden = actions_hidden_states[b:b + 1]  # (1, T, N, D)
#             if len(proprio.shape) == 1:
#                 proprio = proprio.unsqueeze(0)
#             sample_proprio = proprio[b:b + 1] if proprio is not None else None

#             sample_experts = top_k_indices[b]  # (K,)
#             sample_outputs = []

#             for expert_idx in sample_experts:
#                 expert = self.stage_experts[expert_idx.item()]
#                 expert_out = expert.predict_action(
#                     sample_hidden,  # (1, T, N, D)
#                     proprio=sample_proprio,
#                     proprio_projector=proprio_projector,
#                     phase=phase,
#                     use_moe=True
#                 )
#                 # expert_out shape: (1, seq_len, action_dim)
#                 # ✅ 关键修改：移除batch维度
#                 sample_outputs.append(expert_out.squeeze(0))  # (seq_len, action_dim)

#             # Stack along expert dim: (K, seq_len, action_dim)
#             sample_outputs = torch.stack(sample_outputs, dim=0)
#             expert_outputs.append(sample_outputs)

#         # Final shape: (B, K, seq_len, action_dim)
#         expert_outputs_tensor = torch.stack(expert_outputs, dim=0)
#         return expert_outputs_tensor

#     def predict_action(
#             self,
#             actions_hidden_states,
#             proprio=None,
#             proprio_projector=None,
#             phase="Inference",
#             action_probs=None,  # 外部传入的动作概率 (B, num_action_ids)
#             use_moe=True,
#     ):
#         batch_size = actions_hidden_states.shape[0]
#         training = phase == "Training"

#         if not use_moe:
#             # 不使用MoE，直接用基础模型
#             action = L1RegressionActionHead.predict_action(
#                 actions_hidden_states,
#                 proprio=proprio,
#                 proprio_projector=proprio_projector,
#                 phase=phase,
#                 use_moe=False)
#             return action

#         # ========== 使用外部传入的action_probs ==========
#         if action_probs is None:
#             # 如果没有传入，使用均匀分布
#             stage_probs = torch.ones(
#                 batch_size, self.num_experts,
#                 device=actions_hidden_states.device
#             ) / self.num_experts
#         else:
#             # 聚合action_probs到stage_probs
#             stage_probs = self._aggregate_action_probs_to_stage_probs(action_probs)

#         # ========== 选择Top-K专家 ==========
#         top_k_weights, top_k_indices = torch.topk(
#             stage_probs, self.top_k, dim=-1
#         )  # (B, K), (B, K)

#         # 归一化top-k权重
#         top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-10)
#         top_k_weights = top_k_weights.to(torch.bfloat16)
#         # top_k_indices = top_k_indices.to(torch.bfloat16)
#         # ========== 计算负载均衡损失 ==========
#         if training:
#             load_balance_loss, metrics = self.load_balance_loss_fn(
#                 stage_probs, top_k_indices
#             )
#             self.aux_loss = load_balance_loss
#             self.load_balance_metrics = metrics

#         # ========== 稀疏执行专家 ==========
#         expert_outputs = self._sparse_expert_execution(
#             actions_hidden_states,
#             top_k_indices,
#             proprio=proprio,
#             proprio_projector=proprio_projector,
#             phase=phase
#         )  # (B, K, seq_len, D)
#         # ========== 融合专家输出 ==========
#         # ========== ⭐ MoE 强 gate（关键） ==========
#         expert_gate = top_k_weights  # 语义明确：这是 expert scaling gate
#         expert_outputs = expert_outputs * expert_gate[:, :, None, None]

#         # 提取主专家和辅助专家
#         main_features = expert_outputs[:, 0, :, :]  # (B, seq_len, D)
#         aux_features = expert_outputs[:, 1:, :, :]  # (B, K-1, seq_len, D)
#         # import pdb;pdb.set_trace()
#         # 使用交叉注意力融合c
#         fused_features = self.cross_attn(
#             main_features,
#             aux_features,
#             gate_weights=top_k_weights,
#         )

#         # ========== 预测最终动作 ==========
#         action_output = self.action_projection(fused_features)

#         return action_output

#     def get_aux_loss(self):
#         """获取辅助损失（负载均衡损失）"""
#         return self.aux_loss

#     def get_load_balance_metrics(self):
#         """获取负载均衡的统计指标"""
#         return self.load_balance_metrics
###############################################只执行一个expert#####################
import torch
import torch.nn as nn

# 注意：需要确保 LoadBalanceLoss 已定义
# from your_module import LoadBalanceLoss, L1RegressionActionHead

class StageMoEActionHead(nn.Module):
    """
    最简MoE：判别器 → 阶段概率 → argmax硬路由 → 单专家直接输出动作。
    不使用 CrossExpertAttention、残差修正、或任何专家融合。

    对照实验设计：
        - Baseline: 单一 L1RegressionActionHead
        - Experiment: 本模块（3个独立 L1RegressionActionHead，按阶段路由）
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        use_pro_version=False,
        stage_definitions=None,
        num_action_ids=18,
        load_balance_alpha=0.01,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_action_ids = num_action_ids

        # 阶段定义：action_id → stage 的映射
        if stage_definitions is not None:
            self.stages = stage_definitions
        else:
            self.stages = {
                0: list(range(0, 6)),   # 预接触
                1: list(range(6, 12)),  # 交互
                2: list(range(12, 18)), # 后调整
            }
        self.num_experts = len(self.stages)

        # 每个阶段一个独立的完整 L1RegressionActionHead
        self.stage_experts = nn.ModuleList([
            L1RegressionActionHead(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                use_pro_version=use_pro_version
            ) for _ in range(self.num_experts)
        ])

        # 负载均衡损失（训练时用）
        self.load_balance_loss_fn = LoadBalanceLoss(
            num_experts=self.num_experts,
            alpha=load_balance_alpha,
        )

        self.aux_loss = 0.0
        self.load_balance_metrics = {}
        self.last_selected_expert_idx = None
        self.last_stage_probs = None

    # ------------------------------------------------------------------ #
    #  action_probs (B, num_action_ids) → stage_probs (B, num_experts)
    # ------------------------------------------------------------------ #
    def _aggregate_action_probs_to_stage_probs(self, action_probs):
        B = action_probs.shape[0]
        device = action_probs.device
        stage_probs = torch.zeros(B, self.num_experts, device=device)

        for stage_idx, action_ids in self.stages.items():
            for aid in action_ids:
                if aid < action_probs.shape[1]:
                    stage_probs[:, stage_idx] += action_probs[:, aid]

        # 归一化
        stage_probs = stage_probs / (stage_probs.sum(dim=-1, keepdim=True) + 1e-10)
        return stage_probs

    # ------------------------------------------------------------------ #
    #  核心：硬路由 → 单专家执行
    # ------------------------------------------------------------------ #
    def predict_action(
        self,
        actions_hidden_states,
        proprio=None,
        proprio_projector=None,
        phase="Inference",
        action_probs=None,   # (B, num_action_ids) 判别器输出
        use_moe=True,
    ):
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        training = (phase == "Training")
        
        # 姿态信息重塑
        if proprio is not None:
            proprio = proprio.reshape(batch_size, -1)

        # ---------- 不用 MoE 时直接走第 0 个专家（等价于 baseline） ----------
        if not use_moe:
            return self.stage_experts[0].predict_action(
                actions_hidden_states,
                proprio=proprio,
                proprio_projector=proprio_projector,
                phase=phase,
                use_moe=False,
            )

        # ---------- 计算 stage 概率 ----------
        if action_probs is None:
            # 没有判别器输出 → 均匀分布（回退策略）
            stage_probs = torch.ones(
                batch_size, self.num_experts, device=device
            ) / self.num_experts
        else:
            stage_probs = self._aggregate_action_probs_to_stage_probs(action_probs)

        # ---------- argmax 硬路由：每个样本选 1 个专家 ----------
        selected_expert_idx = torch.argmax(stage_probs, dim=-1)  # (B,)
        self.last_selected_expert_idx = selected_expert_idx.detach()
        self.last_stage_probs = stage_probs.detach()

        # ---------- 训练时计算负载均衡损失 ----------
        if training:
            lb_loss, metrics = self.load_balance_loss_fn(
                stage_probs,
                selected_expert_idx.unsqueeze(-1),  # (B, 1)
            )
            self.aux_loss = lb_loss
            self.load_balance_metrics = metrics

        # ---------- 按样本分发到对应专家 ----------
        action_list = [None] * batch_size

        for expert_idx in range(self.num_experts):
            mask = (selected_expert_idx == expert_idx)
            if not mask.any():
                continue

            indices = mask.nonzero(as_tuple=True)[0]
            expert_hidden = actions_hidden_states[mask]
            expert_proprio = proprio[mask] if proprio is not None else None

            # use_moe=False → L1Head 走完整路径，fc2 直接输出 action_dim
            expert_action = self.stage_experts[expert_idx].predict_action(
                expert_hidden,
                proprio=expert_proprio,
                proprio_projector=proprio_projector,
                phase=phase,
                use_moe=False,
            )

            for i, idx in enumerate(indices):
                action_list[idx.item()] = expert_action[i]

        return torch.stack(action_list, dim=0)

    # ------------------------------------------------------------------ #
    #  训练时用：获取辅助损失与负载均衡指标
    # ------------------------------------------------------------------ #
    def get_aux_loss(self):
        return self.aux_loss

    def get_load_balance_metrics(self):
        return self.load_balance_metrics

    def get_last_routing_info(self):
        return {
            "selected_expert_idx": self.last_selected_expert_idx,
            "stage_probs": self.last_stage_probs,
        }

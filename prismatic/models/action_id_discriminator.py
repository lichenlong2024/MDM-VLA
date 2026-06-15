# prismatic/models/action_id_discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionFusion(nn.Module):
    """使用可学习查询向量与task/action特征进行交叉注意力计算"""

    def __init__(self, embed_dim, num_heads=8, num_queries=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_dim = embed_dim // num_heads

        # 可学习的查询向量
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))

        # QKV投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # LayerNorm和Dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, task_features, action_features):
        """
        Args:
            task_features: 任务特征 [B, T, D]
            action_features: 动作特征 [B, A, D]
        """
        batch_size = task_features.shape[0]

        # 拼接所有输入特征
        combined_features = torch.cat([task_features, action_features], dim=1)

        # 扩展查询向量到批次大小
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        # 计算Q, K, V
        q = self.q_proj(queries)
        k = self.k_proj(combined_features)
        v = self.v_proj(combined_features)

        # 多头分割
        q = q.view(batch_size, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 应用注意力权重
        attended_values = torch.matmul(attn_weights, v)

        # 合并多头
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, self.num_queries, self.embed_dim)

        # 输出投影
        output = self.out_proj(attended_values)
        output = self.norm(output)
        # 池化为单个向量
        fused_output = output.mean(dim=1)

        return fused_output


class ActionIDDiscriminator(nn.Module):
    def __init__(self, num_action_ids: int, vision_backbone=None, proprio_dim=8, hidden_dim: int = 512,
                 llm_dim: int = 2048):
        """
        Action ID判别器，用于预测动作类别

        Args:
            num_action_ids: 动作类别数量
            vision_backbone: 视觉骨干网络
            proprio_dim: 本体状态维度
            hidden_dim: 隐藏层维度
            llm_dim: LLM的维度
        """
        super().__init__()

        self.llm_dim = llm_dim
        self.hidden_dim = hidden_dim

        # 视觉骨干网络
        self.vision_backbone = vision_backbone
        if self.vision_backbone is not None:
            self.visual_projector = nn.Sequential(
                nn.Linear(self.vision_backbone.embed_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
        else:
            # 如果没有视觉骨干，假设视觉特征已直接输入（根据你的逻辑，这里可能不需要）
            # 这里做一个保险，实际你的逻辑中vision_backbone应该总是存在的
            self.visual_projector = None

        # 本体状态编码器
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

        # Task/Action特征处理器
        self.task_feature_processor = nn.Sequential(
            nn.Linear(self.llm_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.action_feature_processor = nn.Sequential(
            nn.Linear(self.llm_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # 交叉注意力融合模块
        self.cross_attention_fusion = CrossAttentionFusion(
            embed_dim=hidden_dim,
            num_heads=8,
            num_queries=4
        )

        # 特征融合层 (视觉 + 交叉注意力融合特征 + 本体状态)
        # 注意：这里去掉了 text_features，因为我们不再使用内部的 text_encoder
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 256, hidden_dim),  # visual (h_d) + fused_ta (h_d) + proprio (256)
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )

        # 分类头
        self.classifier = nn.Linear(hidden_dim, num_action_ids)

        # 置信度头
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, pixel_values=None, proprio=None, input_ids=None, task_features=None, action_features=None):
        """
        前向传播 (已修复参数名和 dtype 问题)

        Args:
            pixel_values: 输入图像张量或字典
            proprio: 本体状态信息
            input_ids: 语言指令的token IDs (已不再使用)
            task_features: VLM输出的task特征 [B, T, D]
            action_features: VLM输出的action特征 [B, A, D]

        Returns:
            action_probs: 动作概率分布
            confidence: 置信度
            logits: 分类logits
        """
        # --- 1. 处理视觉输入 ---
        # --- 1. 处理视觉输入 (修复双视角12通道逻辑) ---
        visual_features = None
        if self.vision_backbone is not None and pixel_values is not None:
            # 处理 DinoSigLIP 字典格式（如果输入已提前拆分）
            if isinstance(pixel_values, dict) and "dino" in pixel_values and "siglip" in pixel_values:
                image_dict = pixel_values
            # 处理单视角6通道张量 (DINO:3 + SigLIP:3)
            elif len(pixel_values.shape) == 4 and pixel_values.shape[1] == 6:
                dino_img = pixel_values[:, :3, :, :]  # 前3通道：DINO
                siglip_img = pixel_values[:, 3:, :, :]  # 后3通道：SigLIP
                image_dict = {"dino": dino_img, "siglip": siglip_img}
            # 处理双视角12通道张量 (视角1_DINO:3 + 视角1_SigLIP:3 + 视角2_DINO:3 + 视角2_SigLIP:3)
            elif len(pixel_values.shape) == 4 and pixel_values.shape[1] == 12:
                # 提取两个视角的DINO输入（前3通道 + 中间后3通道）
                dino_img_view1 = pixel_values[:, :3, :, :]
                dino_img_view2 = pixel_values[:, 6:9, :, :]
                # 提取两个视角的SigLIP输入（中间3通道 + 最后3通道）
                siglip_img_view1 = pixel_values[:, 3:6, :, :]
                siglip_img_view2 = pixel_values[:, 9:, :, :]
                # 合并两个视角的DINO和SigLIP（在batch维度拼接，不改变空间维度）
                # 合并后形状：[B*2, 3, H, W]（B为原batch size，2为双视角）
                dino_img_combined = torch.cat([dino_img_view1, dino_img_view2], dim=0)
                siglip_img_combined = torch.cat([siglip_img_view1, siglip_img_view2], dim=0)
                image_dict = {"dino": dino_img_combined, "siglip": siglip_img_combined}
            else:
                raise ValueError(f"Unsupported pixel_values shape: {pixel_values.shape} (expected 6 or 12 channels)")

            # 传入视觉骨干网络计算特征
            visual_features = self.vision_backbone(image_dict)

            # 处理双视角合并后的特征（将[B*2, D]的特征拆分为[B, 2*D]，与单视角特征维度对齐）
            if pixel_values.shape[1] == 12:
                batch_size = pixel_values.shape[0]
                # 拆分双视角特征：[B*2, D] -> [B, 2, D]
                visual_features = visual_features.view(batch_size, 2, -1)
                # 合并双视角特征（在特征维度拼接，最终形状：[B, 2*D]）
                visual_features = visual_features.mean(dim=1)  # 或用torch.cat(visual_features.unbind(dim=1), dim=-1)

            # 确保视觉特征是[B, D]的二维张量（单视角：[B, D]；双视角：[B, 2*D]）
            if len(visual_features.shape) > 2:
                visual_features = visual_features.mean(dim=1)

            # 投影视觉特征到hidden_dim（与其他特征维度对齐）
            visual_features = self.visual_projector(visual_features)

        # --- 2. 处理本体状态 (核心 dtype 修复) ---
        proprio_features = None
        if proprio is not None:
            # 强制将输入 proprio 转换为与 encoder 权重相同的 dtype (bfloat16)
            # 这是解决 RuntimeError 的关键
            proprio = proprio.to(dtype=self.proprio_encoder[0].weight.dtype)
            proprio_features = self.proprio_encoder(proprio)
            batch_size = visual_features.shape[0] if visual_features is not None else task_features.shape[0]
        else:
            # 如果 proprio 为 None，生成一个零向量作为特征
            batch_size = visual_features.shape[0] if visual_features is not None else task_features.shape[0]
            device = visual_features.device if visual_features is not None else task_features.device
            proprio_features = torch.zeros(batch_size, 256, device=device, dtype=self.proprio_encoder[-1].weight.dtype)

        # --- 3. 处理 Task/Action 特征并融合 ---
        # 强制转换 task_features 和 action_features 的 dtype
        if task_features is not None:
            task_features = task_features.to(dtype=self.task_feature_processor[0].weight.dtype)
        if action_features is not None:
            action_features = action_features.to(dtype=self.action_feature_processor[0].weight.dtype)
        if proprio_features.dim() == 1:
            proprio_features = proprio_features.unsqueeze(0)
        processed_task = self.task_feature_processor(task_features)
        processed_action = self.action_feature_processor(action_features)
        fused_ta_features = self.cross_attention_fusion(processed_task, processed_action)
        # --- 4. 融合所有特征 ---
        # 确保所有待拼接特征的 dtype 一致
        dtype = self.feature_fusion[0].weight.dtype
        #import pdb ;pdb.set_trace()
        combined_features = torch.cat([
            visual_features.to(dtype),
            fused_ta_features.to(dtype),
            proprio_features.to(dtype)
        ], dim=-1)
        fused_features = self.feature_fusion(combined_features)

        # --- 5. 输出结果 ---
        logits = self.classifier(fused_features)
        action_probs = F.softmax(logits, dim=-1)
        confidence = self.confidence_head(fused_features).squeeze(-1)

        return action_probs, confidence, logits


class ActionIDLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        # CrossEntropyLoss 默认在 Float32 下工作最好
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets, confidence):
        # --- 关键修改 ---
        #import  pdb;pdb.set_trace()
        # 将 logits 从 BFloat16 转换为 Float32 以计算损失
        logits = logits.to(torch.float32)
        targets = targets.to(torch.long)
        if targets.dim() > 1:
            targets = targets.squeeze(-1)

        # 现在 cross_entropy 可以正常工作了
        classification_loss = self.ce_loss(logits, targets)

        # 计算概率时，也使用转换后的 logits
        probs = F.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]

        # 确保 confidence 和 max_probs 数据类型一致
        confidence_loss = F.mse_loss(confidence.to(torch.float32), max_probs)

        total_loss = classification_loss + self.alpha * confidence_loss
        return total_loss
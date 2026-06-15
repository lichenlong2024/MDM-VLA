#!/usr/bin/env python3
"""
测试改进后的Action ID判别器
"""

import torch
import torch.nn as nn
from prismatic.models.action_id_discriminator import ActionIDDiscriminator

def test_action_id_discriminator():
    # 模拟VLM模型结构
    class MockVLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_backbone = MockVisionBackbone()
            self.llm_dim = 2048
            self.vocab_size = 32000
    
    class MockVisionBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 1024
            
        def forward(self, x):
            # 返回模拟的视觉特征
            batch_size = x["dino"].shape[0] if isinstance(x, dict) else x.shape[0]
            return torch.randn(batch_size, self.embed_dim)
    
    # 创建模拟VLM
    mock_vlm = MockVLM()
    
    # 创建Action ID判别器
    num_action_ids = 10
    discriminator = ActionIDDiscriminator(
        num_action_ids=num_action_ids,
        vlm=mock_vlm,
        hidden_dim=512
    )
    
    # 测试数据
    batch_size = 4
    images = {
        "dino": torch.randn(batch_size, 3, 224, 224),
        "siglip": torch.randn(batch_size, 3, 224, 224)
    }
    proprio = torch.randn(batch_size, 8)
    input_ids = torch.randint(0, 32000, (batch_size, 50))  # 50个token的指令
    
    # 测试前向传播
    action_probs, confidence, logits = discriminator(images, proprio, input_ids=input_ids)
    
    print(f"Input shapes:")
    print(f"  Images: {images['dino'].shape}")
    print(f"  Proprio: {proprio.shape}")
    print(f"  Input IDs: {input_ids.shape}")
    
    print(f"Output shapes:")
    print(f"  Action probs: {action_probs.shape}")
    print(f"  Confidence: {confidence.shape}")
    print(f"  Logits: {logits.shape}")
    
    # 验证输出是否符合预期
    assert action_probs.shape == (batch_size, num_action_ids), f"Action probs shape mismatch: {action_probs.shape}"
    assert confidence.shape == (batch_size,), f"Confidence shape mismatch: {confidence.shape}"
    assert logits.shape == (batch_size, num_action_ids), f"Logits shape mismatch: {logits.shape}"
    
    # 验证概率和为1
    prob_sums = action_probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums)), "Action probabilities do not sum to 1"
    
    # 验证置信度在[0,1]范围内
    assert torch.all(confidence >= 0) and torch.all(confidence <= 1), "Confidence values out of range [0,1]"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_action_id_discriminator()
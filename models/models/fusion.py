"""门控融合层"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class GateFusion(nn.Module):
    """GATE自适应门控融合层（融合图特征和图像特征）"""
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.gate_layer = nn.Linear(hidden_dim, hidden_dim)
        self.latest_gate_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        graph_feat: torch.Tensor,
        img_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if graph_feat.shape != img_feat.shape:
            raise ValueError(
                f"graph_feat and img_feat shape mismatch: {graph_feat.shape} vs {img_feat.shape}"
            )

        # 逐元素相加 + 门控权重计算
        sum_feat = graph_feat + img_feat
        gate = torch.sigmoid(self.gate_layer(sum_feat))

        # 自适应融合
        fused = gate * graph_feat + (1.0 - gate) * img_feat
        self.latest_gate_weights = gate
        return fused, gate

    def get_gate_weights(self) -> Optional[torch.Tensor]:
        """返回门控融合权重"""
        return self.latest_gate_weights

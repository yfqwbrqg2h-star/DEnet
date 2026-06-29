"""预测头（多任务FCN）"""
import torch
import torch.nn as nn
from typing import Sequence, List


class MultiTaskFCNHead(nn.Module):
    """多任务全连接预测头"""
    def __init__(
        self,
        hidden_dim: int = 128,
        fcn_hidden_dims: Sequence[int] = (256, 128),
        num_tasks: int = 19,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 共享FCN层
        layers: List[nn.Module] = []
        in_dim = hidden_dim
        for out_dim in fcn_hidden_dims:
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            in_dim = out_dim
        self.shared_fcn = nn.Sequential(*layers)

        # 多任务独立输出头
        self.task_heads = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(num_tasks)])

    def forward(self, fused_feat: torch.Tensor) -> torch.Tensor:
        h = self.shared_fcn(fused_feat)
        task_outputs = [head(h) for head in self.task_heads]
        return torch.cat(task_outputs, dim=-1)

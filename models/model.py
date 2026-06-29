"""总模型：Knowledge-Augmented GAT"""
import torch
import torch.nn as nn
from typing import Union, Tuple, Dict, Optional, Sequence

from .encoders import MolecularGraphEncoder, MolecularImageEncoder
from .fusion import GateFusion
from .heads import MultiTaskFCNHead
from knowledge_aug_gat.utils.qm9_utils import extract_image_batch
from torch_geometric.data import Data


class KnowledgeAugGAT(nn.Module):
    """Knowledge-Augmented Graph Attention Network 总模型"""
    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int = 128,
        num_tasks: int = 19,
        num_attention_heads: int = 4,
        num_gcn_layers: int = 2,
        num_gat_layers: int = 2,
        gat_heads: int = 4,
        edge_dim: Optional[int] = None,
        fcn_hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()

        # 分支A：图编码器
        self.graph_encoder = MolecularGraphEncoder(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            num_gcn_layers=num_gcn_layers,
            num_gat_layers=num_gat_layers,
            gat_heads=gat_heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        # 分支B：图像编码器
        self.image_encoder = MolecularImageEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # 融合层
        self.gate_fusion = GateFusion(hidden_dim=hidden_dim)

        # 预测头
        self.prediction_head = MultiTaskFCNHead(
            hidden_dim=hidden_dim,
            fcn_hidden_dims=fcn_hidden_dims,
            num_tasks=num_tasks,
            dropout=dropout,
        )

    def forward(
        self,
        data: Data,
        return_explain: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, object]]]:
        # 提取输入特征
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") else None
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        img = extract_image_batch(data).to(x.device)

        # 分支A：图特征
        graph_feat = self.graph_encoder(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        
        # 分支B：图像特征
        img_feat = self.image_encoder(img)
        
        # 融合层
        fused_feat, gate_weights = self.gate_fusion(graph_feat, img_feat)
        
        # 预测头
        pred = self.prediction_head(fused_feat)

        if not return_explain:
            return pred

        # 可解释性输出
        aux = {
            "graph_feat": graph_feat,
            "img_feat": img_feat,
            "fused_feat": fused_feat,
            "mha_attention_weights": self.graph_encoder.get_mha_attention_weights(),
            "gat_attention_weights": self.graph_encoder.get_gat_attention_weights(),
            "gate_weights": gate_weights,
        }
        return pred, aux

    # 可解释性接口
    def get_gat_attention_weights(self):
        return self.graph_encoder.get_gat_attention_weights()

    def get_mha_attention_weights(self):
        return self.graph_encoder.get_mha_attention_weights()

    def get_gate_weights(self):
        return self.gate_fusion.get_gate_weights()

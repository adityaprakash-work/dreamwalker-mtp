# ---INFO-----------------------------------------------------------------------
"""
Encoders for EEG.
"""

__all__ = [
    "GATEncoder",
]


# ---DEPEDENCIES----------------------------------------------------------------
import torch as pt
import torch_geometric as ptg


# ---GAT ENCODER----------------------------------------------------------------
class GATEncoder(pt.nn.Module):
    """
    Graph Attention Network (GAT) encoder.

    Args:
    ----
        - in_channels (int): Number of input channels.
        - hidden_channels (int): Number of hidden channels.
        - num_layers (int): Number of GAT layers.
        - out_channels (int): Number of output channels.
        - dropout (float): Dropout rate.
        - gat_heads (int): Number of GAT heads.
        - n_blocks (int): Number of GAT blocks.

    Returns:
    -------
        pt.Tensor: Encoded data.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 256,
        num_layers: int = 2,
        out_channels: int = 4096,
        dropout: float = 0.2,
        gat_heads: int = 4,
        n_blocks: int = 3,
    ):
        super().__init__()
        self.gat_blocks = pt.nn.ModuleList()
        for b in range(n_blocks):
            ich = in_channels if b == 0 else hidden_channels
            och = hidden_channels if b < n_blocks - 1 else out_channels
            gat = ptg.nn.models.GAT(
                ich,
                hidden_channels,
                num_layers,
                out_channels=och,
                dropout=dropout,
                norm="graph",
                jk="last",
                v2=True,
                heads=gat_heads,
            )
            bnm = pt.nn.BatchNorm1d(och)
            self.gat_blocks.extend((gat, bnm))
        self.inter_block_act = pt.nn.ReLU()

    def forward(self, x: pt.Tensor, edge_index: pt.Tensor) -> pt.Tensor:
        for b in range(0, len(self.gat_blocks), 2):
            gat, bnm = self.gat_blocks[b], self.gat_blocks[b + 1]
            residual = x
            x = gat(x, edge_index)
            x = bnm(x)
            if b not in [0, len(self.gat_blocks) - 2]:
                x = self.inter_block_act(x + residual)
            else:
                x = self.inter_block_act(x)
        return x

# ---INFO-----------------------------------------------------------------------
"""
Loss functions.
"""

__all__ = [
    "ContrastivrSimilarityLoss",
]


# ---DEPENDENCIES---------------------------------------------------------------
import torch as pt


# ---SRC------------------------------------------------------------------------
class ContrastiveSimilarityLoss(pt.nn.Module):
    """
    Contrastive similarity loss function. It computes the cosine similarity
    between the EEG and image embeddings and penalizes the model when the
    similarity between the EEG and a negative image is higher than the
    similarity between the EEG and a positive image.

    L(e1, i1, i2) = max(0, sim(e1, i2) - sim(e1, i1) + margin)

    Args:
    ----
        - threshold (float): Margin threshold.
    """

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold

    def forward(
        self, eeg_emb: pt.Tensor, pos_img_emb: pt.Tensor, neg_img_emb: pt.Tensor
    ) -> pt.Tensor:
        pos_sim = pt.cosine_similarity(eeg_emb, pos_img_emb, dim=-1)
        neg_sim = pt.cosine_similarity(eeg_emb, neg_img_emb, dim=-1)
        loss = pt.relu(neg_sim.mean() - pos_sim.mean() + self.threshold)
        return loss

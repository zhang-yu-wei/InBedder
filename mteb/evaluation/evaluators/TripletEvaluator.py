import logging, os

import torch
import numpy as np
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

logger = logging.getLogger(__name__)

from .Evaluator import Evaluator


class TripletEvaluator(Evaluator):
    def __init__(self, anchor, positive, negative, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.anchor = anchor
        self.positive = positive
        self.negative = negative
        self.batch_size = batch_size
        
        self.cache_dir = kwargs['cache_dir']
        self.filter_words = kwargs['filter_words']
    
    def __call__(self, model):
        logger.info(f"Encoding {len(self.anchor)} * 3 sentences...")
        encode_params = {"batch_size": self.batch_size}
        if self.filter_words is not None:
            encode_params["filter_words"] = self.filter_words
        if self.cache_dir is not None:
            encode_params["cache_dir"] = os.path.join(self.cache_dir, "anchor")
        emb_anc = np.asarray(model.encode(self.anchor, **encode_params))
        if self.cache_dir is not None:
            encode_params["cache_dir"] = os.path.join(self.cache_dir, "positive")
        emb_pos = np.asarray(model.encode(self.positive, **encode_params))
        if self.cache_dir is not None:
            encode_params["cache_dir"] = os.path.join(self.cache_dir, "negative")
        emb_neg = np.asarray(model.encode(self.negative, **encode_params))
        
        logger.info("Evaluating...")
        d_ap = paired_cosine_distances(emb_anc, emb_pos)
        d_an = paired_cosine_distances(emb_anc, emb_neg)
        # d_ap = batch_paired_distances(emb_anc, emb_pos)
        # d_an = batch_paired_distances(emb_anc, emb_neg)

        success = d_ap < d_an

        return {"success_rate": np.sum(success) / len(success)}


def batch_paired_distances(xa, xb, mode="cosine", batch_size=512):
    """
    The sklearn version is too slow
    """

    distances = []
    
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for i in range(0, len(xa), batch_size):
        xa_batch = xa[i:i+batch_size]
        if isinstance(xa_batch, np.ndarray):
            xa_batch = torch.tensor(xa_batch, device="cuda")
        elif isinstance(xa_batch, torch.Tensor):
            xa_batch = xa_batch.to("cuda")

        xb_batch = xb[i:i+batch_size]
        if isinstance(xb_batch, np.ndarray):
            xb_batch = torch.tensor(xb_batch, device="cuda")
        elif isinstance(xb_batch, torch.Tensor):
            xb_batch = xb_batch.to("cuda")
        
        if mode == "cosine":
            cur_dists = (1 - cos(xa_batch, xb_batch)).cpu().tolist()
        elif mode == "euclidean":
            cur_dists = torch.cdist(xa_batch.unsqueeze(1), xb_batch.unsqueeze(1), p=2)
            cur_dists = cur_dists.squeeze().cpu().tolist()
        distances.extend(cur_dists)
    return np.asarray(distances)

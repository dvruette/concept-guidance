from pathlib import Path

import torch
import sklearn.linear_model
import sklearn.decomposition
import tqdm
import numpy as np
import safetensors

from concept_guidance.models.base import BaseProbe
from concept_guidance.models.utils import adaptive_flatten, unflatten, stable_mean


class LogisticProbe(BaseProbe):
    def __init__(
        self,
        normalize: bool = True,
        l2_penalty: float = 1.0,
    ):
        self.normalize = normalize
        self.l2_penalty = l2_penalty

        self.mean_ = None
        self.clfs_ = None
        self.ndims_ = None

    def prepare_inputs(self, xs: torch.Tensor):
        if self.normalize:
            # xs = xs - self.mean_
            finfo = torch.finfo(xs.dtype)
            xs = xs / xs.norm(dim=-1, keepdim=True).clip(finfo.tiny)

        if self.ndims_ == 2:
            xs = xs.reshape(xs.shape[0], -1, xs.shape[-1])

        return xs

    @torch.no_grad()
    def fit(self, X: list[torch.Tensor] | torch.Tensor, y: torch.Tensor, show_progress: bool = True):
        xs, ys = adaptive_flatten(X, y)

        if self.normalize:
            self.mean_ = stable_mean(xs, dim=0).unsqueeze(0)

        if xs.ndim == 4:  # classifier over heads
            self.ndims_ = 2
        elif xs.ndim == 3:
            self.ndims_ = 1
        else:
            raise ValueError(f"Invalid input shape: {xs.shape}")

        xs = self.prepare_inputs(xs)

        self.clfs_ = []
        for i in tqdm.trange(xs.shape[-2], disable=not show_progress, smoothing=0.01):
            clf = sklearn.linear_model.LogisticRegression(
                fit_intercept=False,
                solver="liblinear" if self.normalize else "lbfgs",
                penalty="l2" if self.l2_penalty > 0 else None,
                C=1.0 / self.l2_penalty if self.l2_penalty > 0 else 1.0,
                max_iter=300,
                tol=1e-3,
            )
            clf.fit(xs[:, i].cpu().float().numpy(), ys.cpu().float().numpy())
            self.clfs_.append(clf)

    @torch.no_grad()
    def predict(self, X: list[torch.Tensor] | torch.Tensor):
        if isinstance(X, list):
            orig_shape = [x.shape for x in X]
        else:
            orig_shape = X.shape

        xs = adaptive_flatten(X)
        xs = self.prepare_inputs(xs)

        logits = []
        for i, clf in enumerate(self.clfs_):
            pr = clf.predict_proba(xs[:, i].cpu().float())[:, 1]
            logits.append(torch.from_numpy(pr))
        logits = torch.stack(logits, dim=-1)

        if self.ndims_ == 2:
            logits = logits.reshape(logits.shape[0], *orig_shape[0][1:-1])

        return unflatten(logits, shape=orig_shape)

    def save(self, path: str | Path):
        W = torch.stack([torch.from_numpy(clf.coef_[0]) for clf in self.clfs_], dim=0).to(torch.float32)
        b = torch.tensor([clf.intercept_[0] if isinstance(clf.intercept_, np.ndarray) else clf.intercept_ for clf in self.clfs_], dtype=torch.float32)
        safetensors.torch.save_file({
            "W": W,
            "b": b,
        }, path)

    def get_concept_vectors(self):
        W = torch.stack([torch.from_numpy(clf.coef_[0]) for clf in self.clfs_], dim=0).to(torch.float32)
        return W

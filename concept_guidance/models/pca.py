from pathlib import Path

import torch
import sklearn.linear_model
import sklearn.decomposition
import tqdm
import safetensors

from concept_guidance.models.base import BaseProbe
from concept_guidance.models.utils import adaptive_flatten, unflatten, stable_mean


class PCAProbe(BaseProbe):
    def __init__(
        self,
        normalize: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        self.normalize = normalize
        self.device = device

        self.mean_ = None
        self.direction = None
        self.clfs_ = None
        self.ndims_ = None

    def prepare_inputs(self, xs: torch.Tensor, ys: torch.Tensor | None = None):
        xs = xs.to(self.device)
        if self.normalize:
            xs = xs - self.mean_.to(self.device)
            finfo = torch.finfo(xs.dtype)
            xs = xs / xs.norm(dim=-1, keepdim=True).clip(finfo.tiny)

        if self.ndims_ == 2:
            xs = xs.reshape(xs.shape[0], -1, xs.shape[-1])

        if ys is None:
            return xs.to(self.device)
        else:
            ys = (ys - ys.min()) / (ys.max() - ys.min()) * 2 - 1
            ys = ys.reshape((-1,) + (1,) * (xs.ndim - 1))
            return xs.to(self.device), ys.to(self.device)

    @torch.no_grad()
    def fit(self, X: list[torch.Tensor] | torch.Tensor, y: torch.Tensor):
        xs, ys = adaptive_flatten(X, y)

        if self.normalize:
            self.mean_ = stable_mean(xs, dim=0).unsqueeze(0).cpu()

        if xs.ndim == 4:  # classifier over heads
            self.ndims_ = 2
        elif xs.ndim == 3:
            self.ndims_ = 1
        else:
            raise ValueError(f"Invalid input shape: {xs.shape}")

        xs, ys = self.prepare_inputs(xs, ys)

        ## prepare data for PCA
        pca_xs = xs
        # shuffle
        permutation = torch.randperm(pca_xs.shape[0])
        pca_xs = pca_xs[permutation]
        # take differences
        end_idx = (pca_xs.shape[0] // 2) * 2
        pca_xs = pca_xs[0:end_idx:2] - pca_xs[1:end_idx:2]
        # randomizing directions
        pca_xs *= torch.randint(2, size=(pca_xs.shape[0], 1, 1), device=pca_xs.device) * 2 - 1

        directions = []
        for i in tqdm.trange(xs.shape[-2]):
            pca = sklearn.decomposition.PCA(
                n_components=1,
            )
            pca.fit(pca_xs[:, i].cpu().float().numpy())
            directions.append(torch.from_numpy(pca.components_))
        self.direction_ = torch.cat(directions, dim=0).to(xs.dtype)

        alphas = torch.einsum("ild,ld->il", xs, self.direction_.to(self.device))
        self.clfs_ = []
        for i in tqdm.trange(xs.shape[-2]):
            clf = sklearn.linear_model.LogisticRegression(
                fit_intercept=True,
                penalty=None,
                max_iter=500,
                tol=1e-2,
            )
            clf.fit(alphas[:, i, None].cpu().float().numpy(), ys.view(-1).cpu().float().numpy())
            self.clfs_.append(clf)

    @torch.no_grad()
    def predict(self, X: list[torch.Tensor] | torch.Tensor):
        if isinstance(X, list):
            orig_shape = [x.shape for x in X]
        else:
            orig_shape = X.shape

        xs = adaptive_flatten(X)
        xs = self.prepare_inputs(xs)

        ws = self.direction_.to(self.device, dtype=xs.dtype)
        alphas = torch.einsum("ild,ld->il", xs, ws)

        logits = []
        for i, clf in enumerate(self.clfs_):
            pr = clf.predict_proba(alphas[:, i, None].cpu().float())[:, 1]
            logits.append(torch.from_numpy(pr).to(xs.dtype))
        logits = torch.stack(logits, dim=-1)

        return unflatten(logits, shape=orig_shape)
    
    def save(self, path: str | Path):
        W = self.direction_.to(torch.float32).cpu()
        a = torch.cat([torch.from_numpy(clf.coef_) for clf in self.clfs_], dim=0).to(torch.float32)
        b = torch.cat([torch.from_numpy(clf.intercept_) for clf in self.clfs_], dim=0).to(torch.float32)
        safetensors.torch.save_file({
            "W": W,
            "a": a,
            "b": b,
        }, path)

    def get_concept_vectors(self):
        W = self.direction_.to(torch.float32).cpu()
        return W

import torch


def is_homogeneous(X: list[torch.Tensor] | torch.Tensor):
    if isinstance(X, list):
        return all(x.shape == X[0].shape for x in X)
    else:
        return False

def flatten_homogeneous(X: torch.Tensor, y: torch.Tensor | None = None):
    # X: (n_samples, seq_len, n_layers, n_features)
    xs = X.reshape(-1, *X.shape[2:])
    if y is None:
        return xs
    else:
        ys = y.unsqueeze(-1).expand(-1, X.shape[1]).reshape(-1)
        return xs, ys

def flatten_inhomogeneous(X: list[torch.Tensor], y: torch.Tensor | None = None):
    # X: list[tensor[seq_len, n_layers, n_features]]
    if y is None:
        return torch.cat(X, dim=0)
    else:
        ys = [torch.full(x.shape[:1], fill_value=yi, dtype=x.dtype) for x, yi in zip(X, y)]
        xs = torch.cat(X, dim=0)
        ys = torch.cat(ys, dim=0)
        return xs, ys

def adaptive_flatten(X: list[torch.Tensor] | torch.Tensor, y: torch.Tensor | None = None):
    if isinstance(X, list):
        if is_homogeneous(X):
            xs = torch.stack(X)
            return flatten_homogeneous(xs, y)
        else:
            return flatten_inhomogeneous(X, y)
    else:
        return flatten_homogeneous(X, y)
    

def unflatten(xs: torch.Tensor, shape: tuple[int, ...] | list[tuple[int, ...]]):
    if isinstance(shape, list):
        result = []
        curr_idx = 0
        for orig_shape in shape:
            result.append(xs[curr_idx : curr_idx + orig_shape[0]])
            curr_idx += orig_shape[0]
        return result
    else:
        return xs.reshape(*shape[:2], *xs.shape[1:])


def stable_mean(xs: torch.Tensor, dim: int, chunk_size: int = 1024):
    if xs.shape[0] < chunk_size:
        return xs.to(torch.float32).mean(dim=0).to(xs.dtype)
    else:
        # numerically stable mean
        mean_xs = torch.zeros(xs.shape[:dim] + xs.shape[dim + 1 :], dtype=xs.dtype, device=xs.device)
        count = 0
        for i in range(0, xs.shape[0], chunk_size):
            x = xs[i : i + chunk_size].to(torch.float32)
            if x.shape[0] <= chunk_size:
                mean_xs += x.mean(dim=0)
                count += 1
            else:
                mean_xs += x.sum(dim=0) / chunk_size
                count += x.shape[0] / chunk_size
        mean_xs /= count
        return mean_xs.to(xs.dtype)

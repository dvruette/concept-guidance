import torch

def layer_wise_accuracy(ys_pred: torch.Tensor, ys_true: torch.Tensor):
    accs = [(y_pred > 0.5).eq(y_true).float().mean(dim=0) for y_pred, y_true in zip(ys_pred, ys_true) if y_pred.shape[0] > 0]
    acc = torch.stack(accs, dim=0).mean(dim=0)
    return acc.cpu().numpy()

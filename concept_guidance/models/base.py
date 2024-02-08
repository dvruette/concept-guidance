from abc import ABC, abstractmethod
from pathlib import Path

import torch


class BaseProbe(ABC):
    @abstractmethod
    def fit(self, X: list[torch.Tensor] | torch.Tensor, y: torch.Tensor, show_progress: bool = True):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: list[torch.Tensor] | torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path):
        raise NotImplementedError
    
    @abstractmethod
    def get_concept_vectors(self):
        raise NotImplementedError

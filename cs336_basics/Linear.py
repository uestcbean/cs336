import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype]=None):
        """
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """

        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        w_init = self.initialize_weights(out_features, in_features, factory_kwargs)
        self.weight = nn.Parameter(w_init)

    def initialize_weights(self, out_dim: int, in_dim: int, factory_kwargs: dict) -> torch.Tensor:
        """
            Initialze the weights W using truncated
        """
        W = torch.empty(out_dim, in_dim, **factory_kwargs)
        nn.init.xavier_uniform_(W)
        return W
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

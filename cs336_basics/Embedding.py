import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int | None = None):
        """
            vocab_size: int size of the vocabulary
            d_model: int dimension of the embedding
            padding_idx: int | None = None index of the padding token
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx

        self.weights = torch.empty(vocab_size, d_model)
        nn.init.xavier_uniform_(self.weights)
        self.weights = nn.Parameter(self.weights)

        if padding_idx is not None:
            with torch.no_grad():
                self.weights[padding_idx].fill_(0)

    
    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
            tokens: torch.LongTensor shape (batch_size, seq_len) input token indices
            return: torch.Tensor shape (batch_size, seq_len, d_model) output embeddings
        """
        out = self.weights[tokens]

        if self.padding_idx is not None:
            pad_mask = (tokens == self.padding_idx)[..., None]
            out = out.masked_fill(pad_mask, 0.0)

        return out    
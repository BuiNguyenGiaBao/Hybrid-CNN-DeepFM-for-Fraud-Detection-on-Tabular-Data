import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class FactorizationMachine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, emb: torch.Tensor, values: Optional[torch.Tensor] = None) -> torch.Tensor:
        if values is not None:
            emb = emb * values

        # emb: (B, F, K)
        sum_emb = emb.sum(dim=1)                 # (B, K)
        sum_square = sum_emb * sum_emb           # (B, K)
        square_sum = (emb * emb).sum(dim=1)      # (B, K)
        fm = 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)  # (B, 1)
        return fm


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepFM(nn.Module):
    """DeepFM = Linear + FM(2nd-order) + Deep(MLP). Returns logits.
    """

    def __init__(
        self,
        num_classes: int,
        categorical_cardinalities: Optional[List[int]] = None,
        num_numerical: int = 0,
        embed_dim: int = 16,
        deep_hidden: Optional[List[int]] = None,
        dropout: float = 0.2,
        dense_in_dim: Optional[int] = None,
        dense_num_fields: int = 4,
        use_bias: bool = True,
    ):
        super().__init__()

        deep_hidden = deep_hidden or [128, 64]

        self.num_classes = int(num_classes)
        self.categorical_cardinalities = categorical_cardinalities or []
        self.num_cat = len(self.categorical_cardinalities)
        self.num_num = int(num_numerical)
        self.embed_dim = int(embed_dim)
        self.dense_in_dim = dense_in_dim
        self.dense_num_fields = int(dense_num_fields) if dense_in_dim is not None else 0

        # Linear part
        self.linear_cat = nn.ModuleList([nn.Embedding(card, 1) for card in self.categorical_cardinalities])
        self.linear_num = nn.Linear(self.num_num, 1, bias=False) if self.num_num > 0 else None
        self.linear_dense = nn.Linear(self.dense_in_dim, 1, bias=False) if self.dense_in_dim is not None else None
        self.linear_bias = nn.Parameter(torch.zeros(1)) if use_bias else None

        # FM part
        self.fm = FactorizationMachine()
        self.fm_cat_emb = nn.ModuleList([nn.Embedding(card, self.embed_dim) for card in self.categorical_cardinalities])
        self.fm_num_emb = nn.Parameter(torch.randn(self.num_num, self.embed_dim) * 0.01) if self.num_num > 0 else None
        self.fm_dense_proj = (
            nn.Linear(self.dense_in_dim, self.embed_dim * self.dense_num_fields, bias=False)
            if self.dense_in_dim is not None else None
        )

        # Deep part
        total_fields = self.num_cat + self.num_num + self.dense_num_fields
        if total_fields <= 0:
            raise ValueError("DeepFM requires at least one field.")
        deep_in = self.embed_dim * total_fields
        self.mlp = MLP(deep_in, deep_hidden, dropout=dropout)
        deep_out_dim = 1 if self.num_classes == 2 else self.num_classes
        self.deep_out = nn.Linear(self.mlp.out_dim, deep_out_dim)

    def _build_field_embeddings(
        self,
        cat_x: Optional[torch.Tensor],
        num_x: Optional[torch.Tensor],
        dense_x: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embs: List[torch.Tensor] = []
        vals: List[torch.Tensor] = []

        # Categorical fields
        if self.num_cat > 0:
            if cat_x is None:
                raise ValueError("cat_x is required because categorical_cardinalities is not empty.")
            if cat_x.dim() != 2 or cat_x.size(1) != self.num_cat:
                raise ValueError(f"cat_x must be (B, {self.num_cat}).")

            for j, emb_layer in enumerate(self.fm_cat_emb):
                e = emb_layer(cat_x[:, j])  # (B, K)
                embs.append(e.unsqueeze(1))  # (B, 1, K)
                vals.append(torch.ones(e.size(0), 1, 1, device=e.device, dtype=e.dtype))

        # Numerical fields
        if self.num_num > 0:
            if num_x is None:
                raise ValueError("num_x is required because num_numerical > 0.")
            if num_x.dim() != 2 or num_x.size(1) != self.num_num:
                raise ValueError(f"num_x must be (B, {self.num_num}).")

            e_num = self.fm_num_emb.unsqueeze(0).expand(num_x.size(0), -1, -1)  # (B, N, K)
            embs.append(e_num)
            vals.append(num_x.unsqueeze(-1))  # (B, N, 1)

        # Dense input is split into multiple FM fields so FM can learn interactions.
        if self.dense_in_dim is not None:
            if dense_x is None:
                raise ValueError("dense_x is required because dense_in_dim is set.")
            if dense_x.dim() != 2 or dense_x.size(1) != self.dense_in_dim:
                raise ValueError(f"dense_x must be (B, {self.dense_in_dim}).")

            e_dense = self.fm_dense_proj(dense_x)  # (B, dense_num_fields * K)
            e_dense = e_dense.view(dense_x.size(0), self.dense_num_fields, self.embed_dim)  # (B, Fd, K)
            embs.append(e_dense)
            vals.append(torch.ones(dense_x.size(0), self.dense_num_fields, 1, device=dense_x.device, dtype=dense_x.dtype))

        if len(embs) == 0:
            raise ValueError("At least one of (categorical, numerical, dense) must be provided.")

        field_emb = torch.cat(embs, dim=1)  # (B, F_total, K)
        values = torch.cat(vals, dim=1)     # (B, F_total, 1)
        return field_emb, values

    def _linear_part(
        self,
        cat_x: Optional[torch.Tensor],
        num_x: Optional[torch.Tensor],
        dense_x: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if cat_x is not None:
            batch_size, device = cat_x.size(0), cat_x.device
        elif num_x is not None:
            batch_size, device = num_x.size(0), num_x.device
        elif dense_x is not None:
            batch_size, device = dense_x.size(0), dense_x.device
        else:
            raise ValueError("At least one of cat_x, num_x, dense_x must be provided.")

        out = torch.zeros(batch_size, 1, device=device)

        if self.num_cat > 0 and cat_x is not None:
            for j, emb1 in enumerate(self.linear_cat):
                out = out + emb1(cat_x[:, j])

        if self.linear_num is not None and num_x is not None:
            out = out + self.linear_num(num_x)

        if self.linear_dense is not None and dense_x is not None:
            out = out + self.linear_dense(dense_x)

        if self.linear_bias is not None:
            out = out + self.linear_bias

        return out

    def forward(
        self,
        cat_x: Optional[torch.Tensor] = None,
        num_x: Optional[torch.Tensor] = None,
        dense_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        linear_out = self._linear_part(cat_x, num_x, dense_x)  # (B,1)

        field_emb, values = self._build_field_embeddings(cat_x, num_x, dense_x)  # (B,F,K), (B,F,1)

        fm_out = self.fm(field_emb, values)  # (B,1)

        deep_in = field_emb.reshape(field_emb.size(0), -1)  # (B, F*K)
        deep_h = self.mlp(deep_in)
        deep_out = self.deep_out(deep_h)  # (B,1) or (B,C)

        if self.num_classes == 2:
            return linear_out + fm_out + deep_out
        return linear_out + fm_out + deep_out

# -*- coding: utf-8 -*-
"""
PyTorch translation of PiNet2 (rank=1 or rank=3), non-weighted PIX/Dot only.

This mirrors the TensorFlow reference implementation documented at:
https://teoroo-cmc.github.io/PiNN/master/usage/pinet2/

Key pieces (rank=3 / P3):
- d3 = diff / ||diff||
- p3 initialized as zeros: (n_atoms, 3, 1)
- InvarLayer produces:
    p1 (per-atom invariant update) and i1 (per-pair invariant interaction)
- EquivarLayer uses:
    ix = PIXLayer([ind_2, p3])                       # broadcast atom->pair (j-side)
    ix = ScaleLayer([ix, i1])                        # gate by scalar interaction
    scaled_d3 = ScaleLayer([d3[:, :, None], i1])      # inject bond direction
    ix = ix + scaled_d3
    p3 = IP([ind_2, p3, ix])                         # scatter-add to i-side atoms
    p3 = pp_layer(p3)                                # (no activation, no bias)
    p3_dot = DotLayer(p3)                            # (n_atoms, n_channels)

Torch backend contract:
- forward(tensors) returns per-atom energy contributions (N,) or (N,1).
  Pooling to per-structure energy is handled by PiNetPotentialTorch.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union, Callable, Dict, Any

import torch
import torch.nn as nn

from pinn.networks.pinet_torch import (
    PreprocessLayerTorch,
    CutoffFuncTorch,
    PolynomialBasisTorch,
    GaussianBasisTorch,
    ANNOutputTorch,
    OutLayerTorch,
    ResUpdateTorch,
    PILayerTorch,
    FFLayerTorch,
    IPLayerTorch,
)


class ScaleLayerTorch(nn.Module):
    """Equivariant scaling: X'[.., x, r] = X[.., x, r] * s[.., r]."""

    def forward(self, px: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            px: (..., x, r) tensor
            p1: (..., r) tensor (same leading dim as px)
        Returns:
            scaled: (..., x, r)
        """
        return px * p1[:, None, :]


class PIXLayerTorch(nn.Module):
    """Non-weighted PIXLayer: broadcast atom tensor to pair tensor by taking j-side."""

    def forward(self, ind_2: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ind_2: (n_pairs, 2) long, pairs (i, j)
            px:    (n_atoms, x, r) equivariant property
        Returns:
            ix:    (n_pairs, x, r) = px[j]
        """
        if ind_2.dtype != torch.long:
            ind_2 = ind_2.long()
        j = ind_2[:, 1]
        return px[j]


class DotLayerTorch(nn.Module):
    """Non-weighted DotLayer: einsum('ixr,ixr->ir')."""

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        """
        Args:
            px: (n_atoms, x, r)
        Returns:
            dotted: (n_atoms, r)
        """
        return torch.einsum("ixr,ixr->ir", px, px)


class IPLayerEqTorch(nn.Module):
    """Equivariant IP scatter-add over source atom index i (ind_2[:,0])."""

    def forward(self, ind_2: torch.Tensor, px: torch.Tensor, ix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ind_2: (n_pairs, 2) long, (i, j)
            px:    (n_atoms, x, r) used for shape/device/dtype
            ix:    (n_pairs, x, r) interactions to scatter onto i
        Returns:
            new_px: (n_atoms, x, r)
        """
        if ind_2.dtype != torch.long:
            ind_2 = ind_2.long()
        i = ind_2[:, 0]
        out = torch.zeros_like(px)
        out.index_add_(0, i, ix)
        return out


class InvarLayerTorch(nn.Module):
    """Invariant layer in PiNet2: produces (p1, i1)."""

    def __init__(
        self,
        *,
        pp_nodes: Sequence[int],
        pi_nodes: Sequence[int],
        ii_nodes: Sequence[int],
        n_basis: int,
        activation: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = "tanh",
    ) -> None:
        super().__init__()
        self.pp_pre = FFLayerTorch(pp_nodes, activation=activation, use_bias=True) if len(pp_nodes) else nn.Identity()
        self.pi = PILayerTorch(pi_nodes, n_basis=int(n_basis), activation=activation, use_bias=True)
        self.ii = FFLayerTorch(ii_nodes, activation=activation, use_bias=False)
        self.ip = IPLayerTorch()
        # TF InvarLayer uses a biasless/no-activation PP at the end
        self.pp_post = FFLayerTorch(pp_nodes, activation=None, use_bias=False) if len(pp_nodes) else nn.Identity()

    def forward(
        self, *, ind_2: torch.Tensor, p1: torch.Tensor, basis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ind_2: (n_pairs, 2)
            p1:    (n_atoms, n_prop)
            basis: (n_pairs, n_basis)
        Returns:
            p1_new: (n_atoms, n_prop')
            i1:     (n_pairs, n_outs)  (invariant interaction channels)
        """
        p1_in = self.pp_pre(p1) if not isinstance(self.pp_pre, nn.Identity) else p1
        i1 = self.pi(ind_2, p1_in, basis)
        i1 = self.ii(i1)
        p1_new = self.ip(ind_2, p1_in, i1)
        p1_new = self.pp_post(p1_new) if not isinstance(self.pp_post, nn.Identity) else p1_new
        return p1_new, i1
    


class EquivarLayerTorch(nn.Module):
    """Equivariant (rank=3) layer in PiNet2: produces (p3, dotted_p3)."""

    def __init__(
        self,
        *,
        pp_nodes: Sequence[int],
        n_outs: int,
        activation: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = "tanh",
    ) -> None:
        super().__init__()
        self._i1_proj = None  # created lazily to map i1 -> px_channels
        self.pix = PIXLayerTorch()
        self.scale = ScaleLayerTorch()
        self.ip = IPLayerEqTorch()
        # TF EquivarLayer uses an activation MLP on interactions, but NO activation/bias in pp
        self.ii = FFLayerTorch([int(n_outs)], activation=activation, use_bias=True)
        self.pp = FFLayerTorch(pp_nodes, activation=None, use_bias=False) if len(pp_nodes) else nn.Identity()
        self.dot = DotLayerTorch()

    def forward(
        self,
        *,
        ind_2: torch.Tensor,
        p3: torch.Tensor,
        i1: torch.Tensor,
        d3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ind_2: (n_pairs,2)
            p3:    (n_atoms,3,r)
            i1:    (n_pairs,r)  invariant interaction channels
            d3:    (n_pairs,3)  unit bond directions
        Returns:
            p3_new:     (n_atoms,3,r)
            dotted_p3:  (n_atoms,r)
        """
        # # Gate network on invariant edge features
        i1g = self.ii(i1)  # (n_pairs, r)

        # Ensure gate has same channel count as p3 (r)
        i1g = self._project_i1(i1g, out_dim=p3.shape[-1])        

        ix = self.pix(ind_2, p3)              # (n_pairs,3,r)
        ix = self.scale(ix, i1g)              # gate by scalar channels
        scaled_d3 = self.scale(d3[:, :, None], i1g)  # (n_pairs,3,r)
        ix = ix + scaled_d3

        p3_new = self.ip(ind_2, p3, ix)
        p3_new = self.pp(p3_new) if not isinstance(self.pp, nn.Identity) else p3_new
        dotted = self.dot(p3_new)
        return p3_new, dotted
    
    def _project_i1(self, i1: torch.Tensor, out_dim: int) -> torch.Tensor:
        """Project i1 last-dim to out_dim (created lazily on first use)."""
        if i1.shape[-1] == out_dim:
            return i1
        if self._i1_proj is None or self._i1_proj.in_features != int(i1.shape[-1]) or self._i1_proj.out_features != int(out_dim):
            self._i1_proj = nn.Linear(int(i1.shape[-1]), int(out_dim), bias=True).to(device=i1.device, dtype=i1.dtype)
            self.add_module("i1_proj", self._i1_proj)
        return self._i1_proj(i1)


class GCBlock2Torch(nn.Module):
    """One PiNet2 GCBlock: computes new_tensors dict like TF GCBlock.call()."""

    def __init__(
        self,
        *,
        rank: int,
        pp_nodes: Sequence[int],
        pi_nodes: Sequence[int],
        ii_nodes: Sequence[int],
        n_basis: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.invar = InvarLayerTorch(
            pp_nodes=pp_nodes, pi_nodes=pi_nodes, ii_nodes=ii_nodes, n_basis=n_basis, activation=activation
        )
        if self.rank >= 3:
            self.equivar3 = EquivarLayerTorch(pp_nodes=pp_nodes, n_outs=int(ii_nodes[-1]), activation=activation)

    def forward(self, tensors: Dict[str, torch.Tensor], basis: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            tensors: must contain p1, ind_2, d3 and (if rank>=3) p3
            basis:   (n_pairs, n_basis)
        Returns:
            new_tensors with at least {"p1": ...} and (if rank>=3) {"p3": ...}
        """
        ind_2 = tensors["ind_2"]
        p1_new, i1 = self.invar(ind_2=ind_2, p1=tensors["p1"], basis=basis)

        out: Dict[str, torch.Tensor] = {"p1": p1_new}

        if self.rank >= 3:
            p3_new, _dotted = self.equivar3(ind_2=ind_2, p3=tensors["p3"], i1=i1, d3=tensors["d3"])
            out["p3"] = p3_new

        return out


class PiNet2Torch(nn.Module):
    """Full PiNet2 forward (dict-in, per-atom energy out) for Torch backend."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params = dict(params)
        self.rank = int(self.params.get("rank", 3))
        if self.rank not in (1, 3):
            raise NotImplementedError("Torch PiNet2 currently supports rank=1 or rank=3 only (p3).")

        atom_types = self.params["atom_types"]
        rc = float(self.params["rc"])
        cutoff_type = self.params.get("cutoff_type", "f1")
        basis_type = self.params.get("basis_type", "polynomial")
        n_basis = int(self.params.get("n_basis", 5))
        gamma = float(self.params.get("gamma", 3.0))
        center = self.params.get("center", None)

        pp_nodes = self.params.get("pp_nodes", [8, 8])
        pi_nodes = self.params.get("pi_nodes", [8, 8])
        ii_nodes = self.params.get("ii_nodes", [8, 8])
        out_nodes = self.params.get("out_nodes", [8, 8])
        out_units = int(self.params.get("out_units", 1))
        out_pool = self.params.get("out_pool", False)
        act = self.params.get("act", "tanh")
        depth = int(self.params.get("depth", 3))

        self.depth = depth
        self.preprocess = PreprocessLayerTorch(atom_types, rc)
        self.cutoff = CutoffFuncTorch(rc, str(cutoff_type))

        if str(basis_type).lower() == "polynomial":
            self.basis_fn = PolynomialBasisTorch(n_basis)
        elif str(basis_type).lower() == "gaussian":
            self.basis_fn = GaussianBasisTorch(center=center, gamma=gamma, rc=rc, n_basis=n_basis)
        else:
            raise ValueError(f"Unknown basis_type={basis_type!r}")

        self.gc_blocks = nn.ModuleList(
            [
                GCBlock2Torch(
                    rank=self.rank,
                    pp_nodes=pp_nodes,
                    pi_nodes=pi_nodes,
                    ii_nodes=ii_nodes,
                    n_basis=n_basis,
                    activation=str(act),
                )
                for _ in range(self.depth)
            ]
        )
        self.out_layers = nn.ModuleList(
            [OutLayerTorch(out_nodes, out_units, activation=str(act), use_bias=True) for _ in range(self.depth)]
        )
        self.res_update1 = nn.ModuleList([ResUpdateTorch() for _ in range(self.depth)])
        self.res_update3 = nn.ModuleList([ResUpdateTorch() for _ in range(self.depth)]) if self.rank >= 3 else None
        self.ann_output = ANNOutputTorch(out_pool)

    def forward(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            tensors: dict with keys: coord, elems, ind_1 (and optionally cell/ind_2/...)
        Returns:
            per-atom energies (N,) or (N,1)  (pooling handled by PiNetPotentialTorch)
        """
        tensors = self.preprocess(tensors)
        # PiNetTorch preprocess uses "prop" for scalar features; PiNet2 docs call it "p1".
        # Keep both keys to avoid confusion.
        tensors["p1"] = tensors["prop"]

        # Unit bond directions d3 (TF: diff / ||diff||)
        diff = tensors["diff"]  # (n_pairs,3)
        dnorm = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp_min(1e-12)
        tensors["d3"] = diff / dnorm

        # Init equivariants
        n_atoms = int(tensors["ind_1"].shape[0])
        if self.rank >= 3 and "p3" not in tensors:
            tensors["p3"] = torch.zeros((n_atoms, 3, 1), dtype=tensors["coord"].dtype, device=tensors["coord"].device)

        # Basis
        fc = self.cutoff(tensors["dist"])
        basis = self.basis_fn(tensors["dist"], fc=fc)

        # Output accumulator
        output = torch.zeros((n_atoms, self.out_layers[0].out_units), dtype=tensors["coord"].dtype, device=tensors["coord"].device)

        for i in range(self.depth):
            new_tensors = self.gc_blocks[i](tensors, basis)

            output = self.out_layers[i](tensors["ind_1"], new_tensors["p1"], output)

            tensors["p1"] = self.res_update1[i](tensors["p1"], new_tensors["p1"])
            tensors["prop"] = tensors["p1"]

            if self.rank >= 3:
                assert self.res_update3 is not None
                tensors["p3"] = self.res_update3[i](tensors["p3"], new_tensors["p3"])

        output = self.ann_output(tensors["ind_1"], output)

        # Torch potential wrapper expects per-atom contributions
        return output
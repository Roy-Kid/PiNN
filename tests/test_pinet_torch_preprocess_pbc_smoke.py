import torch
from pinn.networks.pinet_torch import PreprocessLayerTorch


def test_preprocess_mic_pbc_wraps_neighbors() -> None:
    """Atoms across the periodic boundary should be neighbors under MIC."""
    torch.manual_seed(0)

    # One structure with two atoms near opposite sides of a 10 Å box.
    L = 10.0
    rc = 0.5
    coord = torch.tensor([[0.1, 0.0, 0.0],
                          [L - 0.1, 0.0, 0.0]], dtype=torch.float32)
    tensors = {
        "ind_1": torch.zeros((2, 1), dtype=torch.long),
        "elems": torch.tensor([1, 1], dtype=torch.long),
        "coord": coord,
        "cell": torch.tensor([[L, 0.0, 0.0],
                              [0.0, L, 0.0],
                              [0.0, 0.0, L]], dtype=torch.float32),
    }

    pp = PreprocessLayerTorch(atom_types=[1], rc=rc)
    out = pp(tensors)

    # Should find both directed pairs: 0->1 and 1->0
    assert out["ind_2"].shape[0] == 2
    # MIC distance should be 0.2 Å
    assert torch.allclose(out["dist"].sort().values, torch.tensor([0.2, 0.2]), atol=1e-5)
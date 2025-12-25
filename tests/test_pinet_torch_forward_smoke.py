import torch
from pinn.networks.pinet_torch import PiNetTorch

def test_pinet_torch_forward_smoke() -> None:
    torch.manual_seed(0)
    n_atoms = 6
    tensors = {
        "ind_1": torch.zeros((n_atoms, 1), dtype=torch.long),   # one structure
        "elems": torch.tensor([1, 6, 1, 1, 8, 1], dtype=torch.long),
        "coord": torch.randn(n_atoms, 3),
    }
    net = PiNetTorch(depth=2, out_units=1, out_pool=False)
    out = net(tensors)
    assert out.shape == (n_atoms,)

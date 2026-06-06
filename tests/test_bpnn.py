# -*- coding: utf-8 -*-
"""unit tests for bpnn implementation"""
import pytest
import numpy as np
import tensorflow as tf

def _manual_sfs():
    lambd = 1.0
    zeta = 1.0
    eta = 0.01
    Rc = 12.0
    Rs = 0.5

    a = np.array([0., 0., 0.])
    b = np.array([1., 0., 0.])
    c = np.array([1., 1., 0.])
    ab = b-a
    ac = c-a
    bc = c-b
    Rab = np.linalg.norm(ab)
    Rac = np.linalg.norm(ac)
    Rbc = np.linalg.norm(bc)
    cosabc = np.dot(ab, ac)/(Rab*Rac)

    def fcut(R, Rcut):
        return 0.5*(np.cos(np.pi*R/Rcut)+1)
    abc = np.arccos(cosabc) * 180/np.pi

    g2_a = np.exp(-eta*(Rab-Rs))*fcut(Rab, Rc) +\
        np.exp(-eta*(Rac-Rs))*fcut(Rac, Rc)
    g3_a = 2**(1-zeta) *\
        (1+lambd*cosabc)**zeta*np.exp(-eta*(Rab**2+Rac**2+Rbc**2)) *\
        fcut(Rab, Rc)*fcut(Rac, Rc)*fcut(Rbc, Rc)
    g4_a = 2**(1-zeta) *\
        (1+lambd*cosabc)**zeta*np.exp(-eta*(Rab**2+Rac**2)) *\
        fcut(Rab, Rc)*fcut(Rac, Rc)

    return g2_a, g3_a, g4_a


@pytest.mark.forked
def test_sfs():
    # test the BP symmetry functions against manual calculations
    # units in the original runner format is Bohr
    from helpers import get_trivial_runner_ds
    from pinn.networks.bpnn import BPNN
    from pinn.io import sparse_batch

    bohr2ang = 0.5291772109
    dataset = get_trivial_runner_ds().apply(sparse_batch(1))
    sf_spec = [
        {'type': 'G2', 'i': 1, 'j': 'ALL',
         'eta': [0.01/(bohr2ang**2)], 'Rs': [0.5*bohr2ang]},
        {'type': 'G3', 'i': 1, 'j': 8, 'k': 1,
         'eta': [0.01/(bohr2ang**2)], 'lambd': [1.0], 'zeta': [1.0]},
        {'type': 'G4', 'i': 1, 'j': 8, 'k': 1,
         'eta': [0.01/(bohr2ang**2)], 'lambd': [1.0], 'zeta': [1.0]}
    ]
    nn_spec = {8: [35, 35], 1: [35, 35]}
    tensors = next(iter(dataset))
    bpnn = BPNN(sf_spec=sf_spec, nn_spec=nn_spec, rc=12*bohr2ang)
    tensors = bpnn.preprocess(tensors)
    g2_a, g3_a, g4_a = _manual_sfs()
    assert np.allclose(tensors['fp_0'][0], g2_a, rtol=5e-3)
    assert np.allclose(tensors['fp_1'][0], g3_a, rtol=5e-3)
    assert np.allclose(tensors['fp_2'][0], g4_a, rtol=5e-3)

# --- shared fixtures for the jacobian tests ---------------------------------
#
# `use_jacobian` swaps the route by which BPNN obtains the force F = -dE/dR.
# Both routes evaluate the same chain rule
#
#     dE/dR = (dE/dfp) . (dfp/dR)
#
# but differ in how the second factor -- the symmetry-function *jacobian*
# dfp/dR -- is supplied:
#   * use_jacobian=True  : dfp/dR is computed analytically up front and
#                          re-attached through a custom gradient (`_fake_fp`),
#                          so backprop never differentiates the SF graph;
#   * use_jacobian=False : the full SF computation is left in the tape and
#                          differentiated by autodiff.
# The two must therefore agree exactly (up to float32 rounding), and both must
# agree with finite differences. The tests below pin down each of these
# mathematical invariants without relying on RNG seeding to make two separately
# initialised models share weights (TF >= 2.15 no longer guarantees that).

_JACOB_SF_SPEC = [
    {'type': 'G2', 'i': 1, 'j': 1, 'Rs': [1., 2.], 'eta': [0.1, 0.5]},
    {'type': 'G2', 'i': 8, 'j': 1, 'Rs': [1., 2.], 'eta': [0.1, 0.5]},
    {'type': 'G2', 'i': "ALL", 'j': "ALL", 'Rs': [1., 2.], 'eta': [0.1, 0.5]},
    {'type': 'G2', 'i': "ALL", 'j': 1, 'Rs': [1.], 'eta': [0.01]},
    {'type': 'G3', 'i': 1, 'j': 8, 'lambd': [0.5, 1.], 'zeta': [1., 2.], 'eta': [0.1, 0.2]},
    {'type': 'G3', 'i': "ALL", 'j': 8, 'lambd': [0.5, 1.], 'zeta': [1., 2.], 'eta': [0.1, 0.2]},
    {'type': 'G4', 'i': 8, 'j': 8, 'lambd': [0.5, 1.], 'zeta': [1., 2.], 'eta': [0.1, 0.2]},
    {'type': 'G4', 'i': 8, 'j': 8, 'k': 1, 'lambd': [0.5, 1.], 'zeta': [1., 2.], 'eta': [0.1, 0.2]},
]
_JACOB_NN_SPEC = {8: [32, 32], 1: [32, 32]}


def _jacob_water():
    """A deterministic, slightly perturbed 2x2x2 water box for jacobian tests."""
    from ase.collections import g2
    water = g2['H2O']
    water.set_cell([3.1, 3.1, 3.1])
    water.set_pbc(True)
    water = water.repeat([2, 2, 2])
    rng = np.random.default_rng(0)  # fixed seed -> reproducible geometry
    pos = water.get_positions()
    water.set_positions(pos + rng.uniform(0, 0.2, pos.shape))
    return water


def _jacob_tensors(atoms, coord=None):
    """Build the BPNN input dict; `coord` overrides the watched coordinates."""
    if coord is None:
        coord = tf.constant(atoms.positions, tf.float32)
    return {
        "coord": coord,
        "ind_1": tf.zeros_like(atoms.numbers[:, np.newaxis], tf.int32),
        "elems": tf.constant(atoms.numbers, tf.int32),
        "cell":  tf.constant(atoms.cell[np.newaxis, :, :], tf.float32),
    }


@pytest.mark.forked
def test_jacob_bpnn():
    """use_jacobian=True and the full autodiff path give identical forces.

    Weights are copied across the two models (`set_weights`) so the comparison
    isolates the jacobian contraction from weight initialisation. This is the
    chain-rule identity dE/dR = (dE/dfp).(dfp/dR), so the agreement is tight.
    """
    from pinn.networks.bpnn import BPNN

    water = _jacob_water()

    def force(model):
        tensors = _jacob_tensors(water)
        with tf.GradientTape() as g:
            g.watch(tensors['coord'])
            en = model(tensors)
        return -g.gradient(en, tensors['coord'])

    bpnn_jacob = BPNN(_JACOB_SF_SPEC, _JACOB_NN_SPEC, use_jacobian=True)
    bpnn_plain = BPNN(_JACOB_SF_SPEC, _JACOB_NN_SPEC, use_jacobian=False)

    # build both models' variables, then make them share weights
    frc_jacob = force(bpnn_jacob)
    force(bpnn_plain)
    bpnn_plain.set_weights(bpnn_jacob.get_weights())
    frc_plain = force(bpnn_plain)

    assert np.allclose(frc_jacob, frc_plain, atol=1e-4, rtol=1e-4)


@pytest.mark.forked
def test_jacob_force_finite_difference():
    """The analytic force -dE/dR matches central finite differences of E(R).

    This validates that the jacobian route yields the *true* gradient of the
    energy, not merely a value consistent with the autodiff route.
    """
    from pinn.networks.bpnn import BPNN

    water = _jacob_water()
    bpnn = BPNN(_JACOB_SF_SPEC, _JACOB_NN_SPEC, use_jacobian=True)

    def energy(positions):
        coord = tf.constant(positions, tf.float32)
        return float(tf.reduce_sum(bpnn(_jacob_tensors(water, coord))))

    tensors = _jacob_tensors(water)
    with tf.GradientTape() as g:
        g.watch(tensors['coord'])
        en = bpnn(tensors)
    frc = (-g.gradient(en, tensors['coord'])).numpy()

    pos = water.get_positions()
    h = 2e-3
    # spot-check a spread of (atom, axis) components; full FD is 24*3 evals
    for atom, axis in [(0, 0), (3, 1), (7, 2), (12, 0), (19, 1), (23, 2)]:
        p_plus = pos.copy();  p_plus[atom, axis] += h
        p_minus = pos.copy(); p_minus[atom, axis] -= h
        fd = -(energy(p_plus) - energy(p_minus)) / (2 * h)
        assert np.isclose(frc[atom, axis], fd, atol=5e-3, rtol=3e-2), (
            f"force[{atom},{axis}]={frc[atom, axis]:.6f} vs FD={fd:.6f}")


@pytest.mark.forked
def test_jacob_matrix_finite_difference():
    """Direct check of the symmetry-function jacobian dfp/dR.

    A *fixed* linear read-out over the fingerprints (no trainable weights) makes
    g = sum_k w_k * fp_k a deterministic function of the coordinates, so its
    gradient dg/dR is exactly the SF jacobian contracted with the known w. We
    check that the analytic-jacobian route, the autodiff route, and central
    finite differences all agree -- isolating the jacobian matrix itself from
    the neural network.
    """
    from pinn.networks.bpnn import BPNN

    water = _jacob_water()

    def readout(tensors, use_jacobian):
        # preprocess + fingerprint carry no trainable variables, so the result
        # is independent of model initialisation.
        bpnn = BPNN(_JACOB_SF_SPEC, _JACOB_NN_SPEC, use_jacobian=use_jacobian)
        t = bpnn.fingerprint(bpnn.preprocess(tensors))
        total = 0.0
        for elem, fp in sorted(t['elem_fps'].items()):
            w = tf.cast(tf.range(1, fp.shape[-1] + 1), fp.dtype) / 10.0
            total = total + tf.reduce_sum(fp * w)
        return total

    def grad(use_jacobian):
        tensors = _jacob_tensors(water)
        with tf.GradientTape() as g:
            g.watch(tensors['coord'])
            total = readout(tensors, use_jacobian)
        return g.gradient(total, tensors['coord']).numpy()

    g_jacob = grad(use_jacobian=True)
    g_plain = grad(use_jacobian=False)

    # analytic jacobian == full autodiff jacobian (chain-rule identity)
    assert np.allclose(g_jacob, g_plain, atol=1e-4, rtol=1e-4)

    # ... and both equal the finite-difference jacobian
    pos = water.get_positions()
    h = 2e-3
    for atom, axis in [(0, 0), (5, 2), (11, 1), (18, 0), (23, 2)]:
        p_plus = pos.copy();  p_plus[atom, axis] += h
        p_minus = pos.copy(); p_minus[atom, axis] -= h
        f_plus = float(readout(_jacob_tensors(water, tf.constant(p_plus, tf.float32)), False))
        f_minus = float(readout(_jacob_tensors(water, tf.constant(p_minus, tf.float32)), False))
        fd = (f_plus - f_minus) / (2 * h)
        assert np.isclose(g_jacob[atom, axis], fd, atol=5e-3, rtol=3e-2), (
            f"dg/dR[{atom},{axis}]={g_jacob[atom, axis]:.6f} vs FD={fd:.6f}")

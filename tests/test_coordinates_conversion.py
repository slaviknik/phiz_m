"""Преобразование декартовых координат в сферические согласовано с углами ФЭУ."""

import numpy as np

import coordinates


def test_cartesian_to_spherical_matches_pmt_angles(detector_dense):
    det = detector_dense
    r, theta, phi = coordinates.cartesian_to_spherical(det.pmt_centers)

    assert np.allclose(r, det.radius)
    assert np.allclose(theta, det.pmt_theta_sph)
    assert np.allclose(phi, det.pmt_phi_sph)

    # диапазоны углов
    assert theta.min() >= 0.0 and theta.max() <= np.pi
    assert phi.min() >= 0.0 and phi.max() < 2.0 * np.pi

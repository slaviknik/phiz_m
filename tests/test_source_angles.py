"""Сферические углы события согласованы с декартовыми направлениями."""

import numpy as np

import coordinates


def test_generate_angles_matches_directions(source):
    energy = 0.01

    theta, phi = source.generate_angles(energy)
    n = source.photon_number(energy)
    assert theta.shape == (n,)
    assert phi.shape == (n,)
    assert theta.min() >= 0.0 and theta.max() <= np.pi
    assert phi.min() >= 0.0 and phi.max() < 2.0 * np.pi

    # косинус полярного угла равномерен на [-1, 1] — это и есть изотропность
    assert abs(np.cos(theta).mean()) < 0.05

    # углы декартовых направлений восстанавливаются обратным преобразованием
    directions = source.generate_directions(energy)
    r, theta_cart, phi_cart = coordinates.cartesian_to_spherical(directions)
    assert np.allclose(r, 1.0)
    assert np.allclose(np.cos(theta_cart), directions[:, 2])
    assert np.allclose(
        np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * np.pi),
        phi_cart,
    )

"""Обратное преобразование сферических координат в декартовы (round trip)."""

import numpy as np

import coordinates


def test_spherical_to_cartesian_roundtrip(rng):
    points = rng.normal(size=(1000, 3))
    points = points / np.linalg.norm(points, axis=1)[:, None] * 2.0

    r, theta, phi = coordinates.cartesian_to_spherical(points)
    restored = coordinates.spherical_to_cartesian(r, theta, phi)

    assert np.allclose(restored, points)

    # вариант с градусами даёт тот же результат
    _, theta_deg, phi_deg = coordinates.cartesian_to_spherical(points, degrees=True)
    restored_deg = coordinates.spherical_to_cartesian(r, theta_deg, phi_deg, degrees=True)
    assert np.allclose(restored_deg, points)

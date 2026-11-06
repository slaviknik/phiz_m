"""Изотропность излучения: направления равномерно распределены по сфере.

Проверяется, что среднее направление равно нулю, косинус полярного угла
равномерен на ``[-1, 1]`` (критерий χ²), азимут равномерен на ``[0, 2*pi)``.
"""

import numpy as np


def test_directions_are_isotropic(event_directions, chi2_crit_999_dof9):
    directions = event_directions
    n = len(directions)

    # стандартная ошибка среднего по одной координате (σ координаты = 1/sqrt(3))
    sigma_mean = 1.0 / np.sqrt(3.0 * n)

    # средний вектор направлений стремится к нулю
    assert np.abs(directions.mean(axis=0)).max() < 5.0 * sigma_mean

    # косинус полярного угла равномерен на [-1, 1]
    cos_theta = directions[:, 2]
    assert abs(cos_theta.mean()) < 5.0 * sigma_mean
    assert abs(cos_theta.std() - 1.0 / np.sqrt(3.0)) < 0.01

    counts, _ = np.histogram(cos_theta, bins=10, range=(-1.0, 1.0))
    expected = n / 10
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    assert chi2 < chi2_crit_999_dof9, f'χ² = {chi2:.2f} при 9 степенях свободы'

    # азимут равномерен на [0, 2*pi)
    phi = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * np.pi)
    sigma_phi = (np.pi / np.sqrt(3.0)) / np.sqrt(n)
    assert abs(phi.mean() - np.pi) < 5.0 * sigma_phi

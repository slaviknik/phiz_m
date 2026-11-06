"""Шаг между соседними фотоумножителями соответствует плотной упаковке на сфере.

Для равномерного (гексагонального) покрытия сферы при плотности
``rho = N / (4*pi*R^2)`` расстояние между соседями равно
``d = sqrt(2 / (sqrt(3) * rho))``.
"""

import numpy as np


def test_neighbour_distance_matches_ideal_packing(detector_dense):
    det = detector_dense

    # геодезическое расстояние до ближайшего соседа, м
    u = det.pmt_centers / np.linalg.norm(det.pmt_centers, axis=1, keepdims=True)
    cos_gamma = np.clip(u @ u.T, -1.0, 1.0)
    np.fill_diagonal(cos_gamma, -2.0)
    nearest = det.radius * np.arccos(cos_gamma.max(axis=1))

    # идеальная гексагональная упаковка
    rho = det.pmt_number / (4.0 * np.pi * det.radius ** 2)
    d_ideal = float(np.sqrt(2.0 / (np.sqrt(3.0) * rho)))

    assert nearest.min() > 0.75 * d_ideal
    assert nearest.max() < d_ideal
    # разброс шага у спирали Фибоначчи невелик
    assert (nearest.max() - nearest.min()) / d_ideal < 0.15

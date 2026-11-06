"""Фотоумножители распределены по поверхности равномерно.

Равномерное распределение точек по сфере означает равномерное распределение
косинуса полярного угла на ``[-1, 1]`` (равномерность по площади).
"""

import numpy as np


def test_pmt_distribution_is_uniform(detector_dense, chi2_crit_999_dof9):
    det = detector_dense
    cos_theta = np.cos(det.pmt_theta_sph)

    # полярные углы не достигают полюсов (точек ровно на полюсах нет)
    assert cos_theta.min() > -1.0
    assert cos_theta.max() < 1.0

    # среднее и стандартное отклонение равномерного распределения на [-1, 1]
    assert abs(cos_theta.mean()) < 0.01
    assert abs(cos_theta.std() - 1.0 / np.sqrt(3.0)) < 0.01

    # критерий согласия χ² по 10 равным интервалам
    counts, _ = np.histogram(cos_theta, bins=10, range=(-1.0, 1.0))
    expected = len(cos_theta) / 10
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    assert chi2 < chi2_crit_999_dof9, f'χ² = {chi2:.2f} при 9 степенях свободы'

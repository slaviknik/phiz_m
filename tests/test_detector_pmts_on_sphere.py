"""Центры фотоумножителей лежат на поверхности сферического детектора."""

import numpy as np


def test_pmt_centers_are_on_sphere(detector_dense):
    det = detector_dense

    norms = np.linalg.norm(det.pmt_centers, axis=1)
    assert np.allclose(norms, det.radius)

    # радиус-векторы нормированы (единичные направления на ФЭУ)
    unit = det.pmt_centers / norms[:, None]
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)

    # норми внутрь сферы (направлены к центру детектора)
    assert np.allclose(det.pmt_normals, -unit)

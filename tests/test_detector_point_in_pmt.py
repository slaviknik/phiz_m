"""Принадлежность точки поверхности детектора фотоумножителю."""

import numpy as np


def test_point_in_pmt_membership(detector, make_sphere_points):
    det = detector
    centers = det.pmt_centers

    # центр фотоумножителя принадлежит самому себе
    for index in (0, 17, 39):
        assert det.point_in_pmt(centers[index])
        assert index in det.which_pmt(centers[index])

    # точка на противоположной стороне сферы не принадлежит ни одному ФЭУ
    far = -centers[0]
    assert not det.point_in_pmt(far)
    assert det.which_pmt(far).size == 0

    # случайная точка поверхности: попадание в фотокатод согласовано с which_pmt
    for point in make_sphere_points(200):
        assert det.point_in_pmt(point) == (det.which_pmt(point).size > 0)

    # доля поверхности, закрытая фотокатодами: N·(1 − cos α)/2 ≈ 2.5 %
    alpha = np.arctan2(det.pmt_radius, det.radius)
    expected = det.pmt_number * (1.0 - np.cos(alpha)) / 2.0
    assert abs(expected - 0.025) < 1e-3

    covered = np.array([det.point_in_pmt(point)
                        for point in make_sphere_points(20_000, seed=1)])
    assert abs(covered.mean() - expected) < 0.01

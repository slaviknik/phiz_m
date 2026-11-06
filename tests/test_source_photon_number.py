"""Число фотонов линейно по энергии события: N = Y·E."""


def test_photon_number_is_linear_in_energy(source):
    assert source.photon_number(0.0) == 0
    assert source.photon_number(0.5) == 5_000
    assert source.photon_number(1.0) == 10_000
    assert source.photon_number(2.5) == 25_000

    # линейность на произвольном наборе энергий
    for energy in (0.1, 0.37, 1.0, 3.0):
        assert source.photon_number(energy) == round(source.Y * energy)

    # алиас n_photons даёт тот же результат
    assert source.n_photons(1.0) == source.photon_number(1.0)

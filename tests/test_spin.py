import pytest

from qftppy import Spin


def test_construction_and_value():
    assert Spin(1).value() == 1.0
    assert Spin(3, 2).value() == 1.5
    assert float(Spin(1, 2)) == 0.5


def test_invalid_denominator():
    with pytest.raises(ValueError):
        Spin(1, 3)


@pytest.mark.parametrize("value, text", [(0.5, "1/2"), (1.5, "3/2"), (2.0, "2"), (0.0, "0")])
def test_from_float_and_repr(value, text):
    assert repr(Spin.from_float(value)) == text


def test_from_float_rejects_non_half_integer():
    with pytest.raises(ValueError):
        Spin.from_float(0.3)


def test_arithmetic():
    half = Spin(1, 2)
    assert half + half == 1
    assert Spin(3, 2) - half == 1
    assert Spin(1) + 0.5 == Spin(3, 2)
    assert 2 * Spin(3, 2) == 3.0
    assert Spin(3, 2) * 2 == 3.0


def test_comparisons():
    assert Spin(1, 2) < 1
    assert Spin(1) <= 1
    assert Spin(3, 2) > 1
    assert Spin(1) >= Spin(1, 2)
    assert Spin(1, 2) == 0.5

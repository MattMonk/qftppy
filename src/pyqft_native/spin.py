class Spin:
    """
    Handles integral and half-integral spins.
    Mimics the C++ Spin class but optimized for Python.
    """

    def __init__(self, numerator, denominator=1):
        if denominator not in [1, 2]:
            raise ValueError(f"Denominator must be 1 or 2, got {denominator}")
        self._numer = int(numerator)
        self._denom = int(denominator)

    @classmethod
    def from_float(cls, value):
        # 0.5 -> 1/2, 1.0 -> 1/1, etc.
        if abs(value * 2 - round(value * 2)) > 1e-5:
            raise ValueError(f"Value {value} is not integral or half-integral")
        if abs(value - round(value)) < 1e-5:
            return cls(round(value), 1)
        else:
            return cls(round(value * 2), 2)

    def value(self):
        return self._numer / self._denom

    def __float__(self):
        return self.value()

    def __repr__(self):
        if self._denom == 1:
            return str(self._numer)
        return f"{self._numer}/{self._denom}"

    def __add__(self, other):
        val = self.value() + (other.value()
                              if isinstance(other, Spin) else float(other))
        return Spin.from_float(val)

    def __sub__(self, other):
        val = self.value() - (other.value()
                              if isinstance(other, Spin) else float(other))
        return Spin.from_float(val)

    def __mul__(self, other):
        return self.value() * other

    def __rmul__(self, other):
        return self.value() * other

    def __lt__(self, other):
        return self.value() < float(other)

    def __le__(self, other):
        return self.value() <= float(other)

    def __gt__(self, other):
        return self.value() > float(other)

    def __ge__(self, other):
        return self.value() >= float(other)

    def __eq__(self, other):
        return abs(self.value() - float(other)) < 1e-5

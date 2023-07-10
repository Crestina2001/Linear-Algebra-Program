from sympy import Expr, Symbol, Integer, Float, simplify, Eq
from numbers import Number

class MatrixElement:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)



class Element(MatrixElement):
    SUPPORTED_TYPES = (bool, int, float, str, Expr)
    def __init__(self, value):
        if isinstance(value, int) or isinstance(value, bool):
            self.value = Integer(int(value))
        elif isinstance(value, float):
            self.value = Float(value)
        elif isinstance(value, complex):
            self.value = value  # SymPy can handle Python's built-in complex type
        elif isinstance(value, str):
            self.value = Symbol(value)
        elif isinstance(value, Expr):
            self.value = value
        else:
            raise TypeError('Invalid type for Element value')

    def is_zero(self):
        return self.value.is_zero if isinstance(self.value, Expr) else self.value == 0

    def __eq__(self, other):
        if isinstance(other, self.SUPPORTED_TYPES):
            other = Element(other)

        float_tolerance = 1e-10  # small threshold for floating point precision
        result = simplify(self.value - other.value)

        if result == 0:
            return True
        elif isinstance(result, Float) and abs(result) < float_tolerance:
            return True
        else:
            return False

    def __add__(self, other):
        if isinstance(other, self.SUPPORTED_TYPES):
            other = Element(other)
        return Element(simplify(self.value + other.value))

    def __radd__(self, other):
        if isinstance(other, self.SUPPORTED_TYPES):
            other = Element(other)
        return Element(simplify(other.value + self.value))

    def __mul__(self, other):
        if isinstance(other, self.SUPPORTED_TYPES):
            other = Element(other)
        return Element(simplify(self.value * other.value))

    def __rmul__(self, other):
        if isinstance(other, self.SUPPORTED_TYPES):
            other = Element(other)
        return Element(simplify(other.value * self.value))

    def __sub__(self, other):
        if isinstance(other, self.SUPPORTED_TYPES):
            other = Element(other)
        return Element(simplify(self.value - other.value))

    def __isub__(self, other):
        if isinstance(other, self.SUPPORTED_TYPES):
            other = Element(other)
        self.value = simplify(self.value - other.value)
        return self

    def __truediv__(self, other):
        if isinstance(other, self.SUPPORTED_TYPES):
            other = Element(other)
        return Element(simplify(self.value / other.value))

    def __str__(self):
        return str(self.value)


    def __abs__(self):
        return abs(self.value)







class UnknownElement(MatrixElement):
    pass  # implement unknown-specific methods...
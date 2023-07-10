# trial
class VectorSpace:
    def __init__(self, basis):
        self.basis = basis  # List of basis vectors
        self.dimension = len(basis)

    def __str__(self):
        return f"Vector space of dimension {self.dimension}"

    def is_in(self, vector):
        # Check if a vector is in the vector space
        # This could involve solving a system of linear equations
        pass

    def add(self, v1, v2):
        # Add two vectors
        return [a + b for a, b in zip(v1, v2)]

    def scalar_multiply(self, scalar, v):
        # Multiply a vector by a scalar
        return [scalar * a for a in v]

    def linear_combination(self, scalars, vectors):
        # Compute a linear combination of vectors
        combination = [self.scalar_multiply(scalar, vector) for scalar, vector in zip(scalars, vectors)]
        return [sum(x) for x in zip(*combination)]

    def is_basis(self, vectors):
        # Check if a set of vectors forms a basis for the space
        # This could involve checking the rank of the matrix formed by these vectors
        pass

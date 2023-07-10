from element_type import *
from copy import deepcopy
import random

class Matrix:
    def __init__(self, *args):
        if len(args) == 1:  # Initialized with a list
            data = args[0]
            self.rows = len(data)
            self.cols = len(data[0]) if self.rows > 0 else 0
            self.data = [[0]*self.cols for _ in range(self.rows)]
            self.set_data(data)
        elif len(args) == 2:  # Initialized with row and column numbers
            self.rows, self.cols = args
            self.data = [[0]*self.cols for _ in range(self.rows)]
        else:
            raise ValueError("Matrix must be initialized with either a 2D list or row and column numbers")

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if isinstance(idx[0], slice) or isinstance(idx[1], slice):
                # Handle slicing
                row_indices = idx[0] if isinstance(idx[0], slice) else slice(idx[0], idx[0]+1)
                col_indices = idx[1] if isinstance(idx[1], slice) else slice(idx[1], idx[1]+1)
                return Matrix([[self.data[i][j] for j in range(*col_indices.indices(self.cols))] for i in range(*row_indices.indices(self.rows))])
            else:
                # Handle single element access
                return self.data[idx[0]][idx[1]]
        else:
            raise IndexError("Invalid index to matrix")

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            if isinstance(index[0], slice) or isinstance(index[1], slice):
                row_indices = index[0] if isinstance(index[0], slice) else slice(index[0], index[0] + 1)
                col_indices = index[1] if isinstance(index[1], slice) else slice(index[1], index[1] + 1)

                for i, row in enumerate(
                        range(row_indices.start or 0, row_indices.stop or self.rows, row_indices.step or 1)):
                    for j, col in enumerate(
                            range(col_indices.start or 0, col_indices.stop or self.cols, col_indices.step or 1)):
                        self.set_cell(row, col, value[i,j])
            else:
                self.set_cell(*index, value)
        else:
            raise ValueError("Index must be a tuple (row, col) or slices")

    # when *args = row, col, element, set the (row, col) place of the matrix with value
    # when *args = list, reset the whole matrix with list
    def set(self, *args):
        if len(args) == 3:
            row, col, element = args
            self.set_cell(row, col, element)
        elif len(args) == 1 and isinstance(args[0], list):
            data = args[0]
            self.set_data(data)
        else:
            raise ValueError("Invalid arguments to set")

    def set_cell(self, row, col, element):
        if row >= self.rows or col >= self.cols:
            raise IndexError("Matrix index out of range")
        if isinstance(element, (MatrixElement, bool)):
            self.data[row][col] = element
        elif isinstance(element, (int, float, str, Expr)):
            self.data[row][col] = Element(element)
        else:
            raise ValueError("element must be an instance of MatrixElement or a numeric type")

    # set the matrix with data(data shall be a list)
    def set_data(self, data):
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0
        self.data = [[0] * self.cols for _ in range(self.rows)]
        for i, row in enumerate(data):
            for j, cell in enumerate(row):
                self.set_cell(i, j, cell)


    # used when printing the matrix
    def __str__(self):
        cell_width = max(len(str(cell)) for row in self.data for cell in row)
        return "\n".join(" ".join(format(str(cell), f"{cell_width}s") for cell in row) for row in self.data)

    def __eq__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions to compare")
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.set(i, j, self.data[i][j] == other.data[i][j])
        return result


    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions to add")
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.set(i, j, self.data[i][j] + other.data[i][j])
        return result

    def __iadd__(self, other):
        temp = self + other
        self.data = temp.data
        return self

    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions to subtract")
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.set(i, j, self.data[i][j] - other.data[i][j])
        return result

    def __isub__(self, other):
        temp = self - other
        self.data = temp.data
        return self

    def __mul__(self, scalar):
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.set(i, j, self.data[i][j] * scalar)
        return result

    def __truediv__(self, scalar):
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.set(i, j, self.data[i][j] / scalar)
        return result

    def __itruediv__(self, scalar):
        temp = self / scalar
        self.data = temp.data
        return self


    def __matmul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Number of columns in first matrix must match number of rows in second matrix for multiplication")
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                cell_value = 0
                for k in range(self.cols):
                    cell_value += self.data[i][k] * other.data[k][j]
                result.set(i, j, cell_value)
        return result

    # used when printing tuple
    def __repr__(self):
        return self.__str__()

    def find(self, value):
        for i in range(self.rows):
            for j in range(self.cols):
                if(self.data[i][j]==value):
                    return (i,j)
        return None

    def find_nonzero_row(self, col):
        for i in range(col, self.rows):
            if not self.data[i][col].is_zero():
                return i
        return None

    def transpose(self, inplace=False):
        if inplace:
            self.data = [[self.data[j][i] for j in range(len(self.data))] for i in range(len(self.data[0]))]
            self.rows, self.cols = self.cols, self.rows
            return self
        else:
            return Matrix([[self.data[j][i] for j in range(len(self.data))] for i in range(len(self.data[0]))])

    def gaussian_elimination(self, inplace=False):
        matrix_to_use = self if inplace else deepcopy(self)

        h = 0  # pivot row
        k = 0  # pivot column

        while h < matrix_to_use.rows and k < matrix_to_use.cols:
            # Find the k-th pivot
            nonzero_row_index = matrix_to_use.find_nonzero_row(k)
            if nonzero_row_index is None:
                # No pivot in this column, pass to next column
                k += 1
            else:
                # Swap rows
                matrix_to_use[h,:], matrix_to_use[nonzero_row_index,:] = matrix_to_use[nonzero_row_index,:], matrix_to_use[h,:]
                if matrix_to_use[h,k].is_zero():
                    continue
                pivot_value = matrix_to_use[h,k].value

                # Do for all rows below pivot
                for i in range(h + 1, matrix_to_use.rows):
                    if matrix_to_use[i,k].is_zero():
                        continue
                    f = matrix_to_use[i,k].value / pivot_value

                    # Fill with zeros the lower part of pivot column
                    matrix_to_use[i,k] = Element(0)

                    # Do for all remaining elements in current row
                    matrix_to_use[i, k + 1:] -= matrix_to_use[h, k + 1:] * f

                # Normalize the pivot row
                matrix_to_use[h, k:] /= pivot_value

                # Increase pivot row and column
                h += 1
                k += 1

        # Back substitution
        for i in range(matrix_to_use.rows - 1, 0, -1):
            pivot_col = matrix_to_use.find_nonzero_row(i)
            if pivot_col is not None:
                for j in range(i):
                    if not matrix_to_use[j, pivot_col].is_zero():
                        f = matrix_to_use[j, pivot_col].value
                        matrix_to_use[j, pivot_col:] -= matrix_to_use[i, pivot_col:] * f

        return matrix_to_use

    def extend_right(self, other):
        # Check if the number of rows is the same in both matrices
        if self.rows != other.rows:
            raise ValueError("The matrices must have the same number of rows.")

        # Extend each row of the first matrix with the corresponding row of the second matrix
        for row_self, row_other in zip(self.data, other.data):
            row_self.extend(row_other)

        # Update the number of columns
        self.cols += other.cols

    @classmethod
    def identity(cls, size):
        return cls([[1 if i == j else 0 for j in range(size)] for i in range(size)])

    @classmethod
    def zeros(cls, rows, cols):
        return cls([[0 for _ in range(cols)] for _ in range(rows)])

    @classmethod
    def random(cls, rows, cols=None, mu=0, sigma=1):
        """Create a new Matrix with random float elements from a Gaussian distribution."""
        if cols is None:
            cols = rows
        return cls([[Element(random.gauss(mu, sigma)) for _ in range(cols)] for _ in range(rows)])


    @classmethod
    def permutation_matrix(cls, permutation):
        n = len(permutation)
        # Check that the permutation is valid
        if set(permutation) != set(range(n)):
            raise ValueError("Invalid permutation")
        P = [[0]*n for _ in range(n)]
        for i, j in enumerate(permutation):
            P[i][j] = 1
        return cls(P)

    def submatrix(self, start_row, end_row, start_col, end_col):
        return Matrix([row[start_col:end_col] for row in self.data[start_row:end_row]])

    def inverse(self, inplace=False):
        if self.rows != self.cols:
            raise ValueError("Only square matrices are invertible.")

        # Create an augmented matrix [A|I]
        augmented_matrix = self if inplace else deepcopy(self)
        augmented_matrix.extend_right(Matrix.identity(self.rows))

        # Perform Gaussian elimination
        augmented_matrix.gaussian_elimination(inplace=True)
        if ((augmented_matrix[:,:self.cols] == Matrix.identity(self.rows)).find(False)):
            print("The inverse of the given matrix does not exist!")
            return None

        # Extract the right half as the inverse
        inverse = augmented_matrix.submatrix(0, self.cols, self.rows, 2 * self.cols)

        # If inplace, update the original matrix to be its inverse
        if inplace:
            self.data = inverse.data
            return self
        else:
            return inverse


    def lu_decomposition(self):
        if(self.rows!=self.cols):
            raise ValueError("Only square matrices can perform LU decomposition.")

        n = self.rows
        L = Matrix.identity(n)
        U = Matrix.zeros(n, n)

        for i in range(n):
            for j in range(i, n):
                U[i, j] = self[i, j] - sum(L[i, k] * U[k, j] for k in range(i))

            for j in range(i, n):
                if i == j:
                    L[i, i] = 1
                else:
                    L[j, i] = (self[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

        return (L, U)
    # P@A=L@U
    def plu_decomposition(self):
        if(self.rows!=self.cols):
            raise ValueError("Only square matrices can perform PLU decomposition.")

        n = self.rows
        P = Matrix.identity(n)
        matrix_to_use = deepcopy(self) # used for Gaussian Elimination
        for k in range(n):
            # Check the type of elements and raise an error if needed
            for i in range(n):
                if isinstance(self[i, k].value, Expr) and \
                        not(isinstance(self[i, k].value, Integer) or isinstance(self[i, k].value, Float)):
                    raise TypeError('Invalid type for Element value')

            # Find the maximum in the current column
            max_row_index = max(range(k, n), key=lambda i: abs(matrix_to_use[i, k]))

            # Check if the matrix is singular and raise an error if needed
            if matrix_to_use[max_row_index, k].value == 0:
                raise ValueError('LU decomposition cannot be performed: the matrix is singular')

            # Swap rows in P and in the original matrix
            P[k, :], P[max_row_index, :] = P[max_row_index, :], P[k, :]
            matrix_to_use[k, :], matrix_to_use[max_row_index, :] = \
                matrix_to_use[max_row_index, :], matrix_to_use[k, :]

            for i in range(k+1,n):
                multiplier = matrix_to_use[i,k] / matrix_to_use[k,k]
                matrix_to_use[i,:]-=matrix_to_use[k,:]*multiplier
        L, U = (P@self).lu_decomposition()
        return P, L, U

from my_matrix import *
from itertools import permutations

def exercise_04():
    # problem 1
    matrix = Matrix([[1, 3, 0], [2, 4, 0], [2, 0, 1]])
    L, U = matrix.lu_decomposition()
    print("Lower triangular matrix L:")
    print(L)
    print("Upper triangular matrix U:")
    print(U)
    print("check")
    print(L@U==matrix)
    print("inverse of L:")
    print(L.inverse())

    # problem 2
    A = Matrix([['a','a','a','a'],['a','b','b','b'],['a','b','c','c'],['a','b','c','d']])
    L, U = A.lu_decomposition()
    print("Lower triangular matrix L:")
    print(L)
    print("Upper triangular matrix U:")
    print(U)
    print("check")
    print(L@U==A)

    # recitations
    A = Matrix([[1,0, 1],['a','a','a'],['b','b','a']])
    L, U = A.lu_decomposition()
    print("Lower triangular matrix L:")
    print(L)
    print("Upper triangular matrix U:")
    print(U)
    print("check")
    print(L@U==A)

    A = Matrix([[2,2.1, 2],[1,1,0],[1,0,1]])
    L, U = A.lu_decomposition()
    print("Lower triangular matrix L:")
    print(L)
    print("Upper triangular matrix U:")
    print(U)
    print("check")
    print(L@U==A)

def exercise_05():
    matrix = Matrix([[1,'a',3],[2,'b',4]])
    print(matrix)
    print(matrix.transpose())

    A = Matrix([[2, 2, 2, 2], [1, 1, 0, 5],[1,0,1, 7],[3,2,1, 0]])
    P, L, U = A.plu_decomposition()
    print('check:')
    print(P@A==L@U)

    A = Matrix.random(10)
    P, L, U = A.plu_decomposition()
    print('check:')
    print(P@A==L@U)

    # Find a 3 by 3 permutation matrix with P3 = I (but not P = I).
    P = Matrix.permutation_matrix([1,2,0])
    print('permutation matrix is:')
    print(P)
    print("check:")
    print(P@P@P)

    # Find a 4 by 4 permutation P with P^4 != I.

    # List to store permutation matrices
    non_identity_permutations = []

    # Generate all permutations
    for perm in permutations(range(4)):
        P = Matrix.permutation_matrix(list(perm))
        # Compute the fourth power of P
        P4 = P @ P @ P @ P
        # Check if P4 is not the identity matrix
        if (P4 == Matrix.identity(4)).find(False):
            non_identity_permutations.append(P)

    print(f"There are {len(non_identity_permutations)} permutations such that P^4 != I.")
    for matrix in non_identity_permutations:
        print(matrix)
        print('')

if __name__ == "__main__":
    exercise_05()

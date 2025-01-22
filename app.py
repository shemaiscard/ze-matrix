# app.py
import streamlit as st
import numpy as np
import pandas as pd
from scipy.linalg import lu as scipy_lu

# Load custom CSS
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Matrix Operations
def matrix_addition(matrix1, matrix2):
    return np.add(matrix1, matrix2)

def matrix_subtraction(matrix1, matrix2):
    return np.subtract(matrix1, matrix2)

def matrix_multiplication(matrix1, matrix2):
    return np.matmul(matrix1, matrix2)

def matrix_transpose(matrix):
    return np.transpose(matrix)

def matrix_determinant(matrix):
    return np.linalg.det(matrix)

def matrix_inverse(matrix):
    return np.linalg.inv(matrix)

def scalar_multiplication(matrix, scalar):
    return np.multiply(matrix, scalar)

def matrix_eigenvalues_and_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

def matrix_rank(matrix):
    return np.linalg.matrix_rank(matrix)

def matrix_trace(matrix):
    return np.trace(matrix)

def matrix_diagonal(matrix):
    return np.diag(matrix)

def matrix_norm(matrix, ord=None):
    return np.linalg.norm(matrix, ord=ord)

def matrix_power(matrix, power):
    return np.linalg.matrix_power(matrix, power)

def matrix_solve_linear_equation(matrix, b):
    return np.linalg.solve(matrix, b)

def matrix_svd(matrix):
    U, S, V = np.linalg.svd(matrix)
    return U, S, V

def matrix_cholesky_decomposition(matrix):
    return np.linalg.cholesky(matrix)

def matrix_qr_decomposition(matrix):
    return np.linalg.qr(matrix)

def matrix_lu_decomposition(matrix):
    return scipy_lu(matrix)

# Helper function to input a matrix using a table
def input_matrix_table(rows, cols, key_prefix):
    matrix = []
    for i in range(rows):
        col1, col2, col3 = st.columns([0.7, 1, 1])  # Create 3 columns with the middle column being smaller
        row = []
        for j in range(cols):
            with col1:  # Place the input box in the middle column
                cell_value = st.number_input(
                    f"Matrix [{i+1}][{j+1}]",
                    value=0,
                    step=1,
                    key=f"{key_prefix}_{i}_{j}",
                    format="%d"
                )
            row.append(cell_value)
        matrix.append(row)
    return np.array(matrix)

# Educational Mode
def educational_mode():
    st.sidebar.header("Educational Mode")
    sub_tab1, sub_tab2 = st.sidebar.tabs(["Concept Explanations", "Interactive Quizzes"])

    # Concept Explanations
    with sub_tab1:
        st.subheader("Matrix Operations Explained")
        operation = st.selectbox("Select a Matrix Operation", [
            "Matrix Addition", "Matrix Subtraction", "Matrix Multiplication", 
            "Matrix Transpose", "Matrix Determinant", "Matrix Inverse",
            "Scalar Multiplication", "Matrix Power", "Matrix Rank",
            "Matrix Trace", "Matrix Diagonal", "Eigenvalues & Eigenvectors",
            "Solve Linear Equation", "Matrix Norm", "SVD", "Cholesky", "QR", "LU"
        ])

        if operation == "Matrix Addition":
            st.write("""
                **Matrix Addition**:
                - Two matrices can be added if they have the same dimensions.
                - The result is a matrix where each element is the sum of the corresponding elements from the two matrices.
                - Example:
                    ```
                    A = [1 2]    B = [5 6]
                        [3 4]        [7 8]
                    A + B = [1+5 2+6] = [6 8]
                            [3+7 4+8]   [10 12]
                    ```
            """)
            st.image("https://www.mathsisfun.com/algebra/images/matrix-addition.gif", caption="Matrix Addition Visualization")

        elif operation == "Matrix Subtraction":
            st.write("""
                **Matrix Subtraction**:
                - Two matrices can be subtracted if they have the same dimensions.
                - The result is a matrix where each element is the difference of the corresponding elements from the two matrices.
                - Example:
                    ```
                    A = [1 2]    B = [5 6]
                        [3 4]        [7 8]
                    A - B = [1-5 2-6] = [-4 -4]
                            [3-7 4-8]   [-4 -4]
                    ```
            """)
            st.image("https://www.mathsisfun.com/algebra/images/matrix-subtraction.gif", caption="Matrix Subtraction Visualization")

        elif operation == "Matrix Multiplication":
            st.write("""
                **Matrix Multiplication**:
                - Two matrices can be multiplied if the number of columns in the first matrix equals the number of rows in the second matrix.
                - The result is a matrix where each element is the dot product of the corresponding row from the first matrix and column from the second matrix.
                - Example:
                    ```
                    A = [1 2]    B = [5 6]
                        [3 4]        [7 8]
                    A * B = [1*5+2*7 1*6+2*8] = [19 22]
                            [3*5+4*7 3*6+4*8]   [43 50]
                    ```
            """)

        elif operation == "Matrix Transpose":
            st.write("""
                **Matrix Transpose**:
                - The transpose of a matrix is obtained by flipping the matrix over its diagonal.
                - The rows become columns, and the columns become rows.
                - Example:
                    ```
                    A = [1 2 3]
                        [4 5 6]
                    A^T = [1 4]
                          [2 5]
                          [3 6]
                    ```
            """)
            st.image("https://www.mathsisfun.com/algebra/images/matrix-transpose.gif", caption="Matrix Transpose Visualization")

        elif operation == "Matrix Determinant":
            st.write("""
                **Matrix Determinant**:
                - The determinant is a scalar value that can be computed from a square matrix.
                - It is used in solving systems of linear equations, finding the inverse of a matrix, and more.
                - Example:
                    ```
                    A = [1 2]
                        [3 4]
                    det(A) = 1*4 - 2*3 = -2
                    ```
            """)

        elif operation == "Matrix Inverse":
            st.write("""
                **Matrix Inverse**:
                - The inverse of a matrix is a matrix that, when multiplied by the original matrix, gives the identity matrix.
                - Only square matrices with a non-zero determinant have an inverse.
                - Example:
                    ```
                    A = [1 2]
                        [3 4]
                    A^-1 = [-2 1]
                           [1.5 -0.5]
                    ```
            """)
            st.image("https://www.mathsisfun.com/algebra/images/matrix-inverse.gif", caption="Matrix Inverse Visualization")

        elif operation == "Scalar Multiplication":
            st.write("""
                **Scalar Multiplication**:
                - Scalar multiplication involves multiplying every element of a matrix by a scalar (a single number).
                - Example:
                    ```
                    A = [1 2]
                        [3 4]
                    2 * A = [2*1 2*2] = [2 4]
                            [2*3 2*4]   [6 8]
                    ```
            """)

        elif operation == "Matrix Power":
            st.write("""
                **Matrix Power**:
                - The power of a matrix is calculated by multiplying the matrix by itself a specified number of times.
                - Example:
                    ```
                    A = [1 2]
                        [3 4]
                    A^2 = A * A = [1*1+2*3 1*2+2*4] = [7 10]
                                  [3*1+4*3 3*2+4*4]   [15 22]
                    ```
            """)

        elif operation == "Matrix Rank":
            st.write("""
                **Matrix Rank**:
                - The rank of a matrix is the maximum number of linearly independent row or column vectors in the matrix.
                - Example:
                    ```
                    A = [1 2]
                        [3 4]
                    rank(A) = 2
                    ```
            """)

        elif operation == "Matrix Trace":
            st.write("""
                **Matrix Trace**:
                - The trace of a matrix is the sum of the elements on the main diagonal (from the top-left to the bottom-right).
                - Example:
                    ```
                    A = [1 2]
                        [3 4]
                    trace(A) = 1 + 4 = 5
                    ```
            """)

        elif operation == "Matrix Diagonal":
            st.write("""
                **Matrix Diagonal**:
                - The diagonal of a matrix is the set of elements where the row index equals the column index.
                - Example:
                    ```
                    A = [1 2]
                        [3 4]
                    diagonal(A) = [1, 4]
                    ```
            """)
        elif operation == "Eigenvalues & Eigenvectors":
            st.write("""
                **Eigenvalues & Eigenvectors**:
                - Eigenvalues and eigenvectors are special numbers and vectors associated with a matrix.
                - They are used in various applications, including stability analysis and quantum mechanics.
                - Example:
                    ```
                    A = [1 2]
                        [3 4]
                    eigenvalues = [ -0.37228132,  5.37228132]
                    eigenvectors = [[-0.82456484, -0.41597356],
                                    [ 0.56576746, -0.90937671]]
                    ```
            """)

        elif operation == "Solve Linear Equation":
            st.write("""
                **Solve Linear Equation**:
                - Solving a linear equation involves finding the vector x that satisfies the equation Ax = b.
                - Example:
                    ```
                    A = [1 2]
                        [3 4]
                    b = [5, 6]
                    x = [-4, 4.5]
                    ```
            """)

        elif operation == "Matrix Norm":
            st.write("""
                **Matrix Norm**:
                - The norm of a matrix is a measure of its size or length.
                - Different types of norms include Frobenius norm, nuclear norm, and more.
                - Example:
                    ```
                    A = [1 2]
                        [3 4]
                    norm(A) = 5.477225575051661
                    ```
            """)

        elif operation == "SVD":
            st.write("""
                **Singular Value Decomposition (SVD)**:
                - SVD is a factorization of a matrix into three matrices: U, S, and V.
                - It is used in various applications, including data compression and noise reduction.
                - Example:
                    ```
                    A = [1 2]
                        [3 4]
                    U = [[-0.40455358, -0.9145143 ],
                         [-0.9145143 ,  0.40455358]]
                    S = [ 5.4649857 ,  0.36596619]
                    V = [[-0.57604844, -0.81741556],
                         [ 0.81741556, -0.57604844]]
                    ```
            """)

        elif operation == "Cholesky":
            st.write("""
                **Cholesky Decomposition**:
                - Cholesky decomposition is a decomposition of a positive definite matrix into a lower triangular matrix and its conjugate transpose.
                - Example:
                    ```
                    A = [1 2]
                        [2 5]
                    L = [1 0]
                        [2 1]
                    ```
            """)

        elif operation == "QR":
            st.write("""
                **QR Decomposition**:
                - QR decomposition is a decomposition of a matrix into an orthogonal matrix and an upper triangular matrix.
                - Example:
                    ```
                    A = [1 2]
                        [3 4]
                    Q = [[-0.31622777, -0.9486833 ],
                         [-0.9486833 ,  0.31622777]]
                    R = [[-3.16227766, -4.42718872],
                         [ 0.        , -0.63245553]]
                    ```
            """)

        elif operation == "LU":
            st.write("""
                **LU Decomposition**:
                - LU decomposition is a decomposition of a matrix into a lower triangular matrix and an upper triangular matrix.
                - Example:
                    ```
                    A = [1 2]
                        [3 4]
                    L = [1 0]
                        [3 1]
                    U = [1 2]
                        [0 -2]
                    ```
            """)

    # Interactive Quizzes
    with sub_tab2:
        st.subheader("Interactive Quizzes")
        quiz_type = st.selectbox("Select a Quiz Type", [
            "Matrix Addition", "Matrix Subtraction", "Matrix Multiplication", 
            "Matrix Transpose", "Matrix Determinant", "Matrix Inverse",
            "Scalar Multiplication", "Matrix Power", "Matrix Rank",
            "Matrix Trace", "Matrix Diagonal", "Eigenvalues & Eigenvectors",
            "Solve Linear Equation", "Matrix Norm", "SVD", "Cholesky", "QR", "LU"
        ])

        if quiz_type == "Matrix Addition":
            st.write("**Quiz: Matrix Addition**")
            st.write("Add the following matrices:")
            A = np.array([[1, 2], [3, 4]])
            B = np.array([[5, 6], [7, 8]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            st.write("B =")
            st.write(pd.DataFrame(B))
            user_answer = input_matrix_table(2, 2, "quiz_add")
            if st.button("Submit"):
                try:
                    correct_answer = matrix_addition(A, B)
                    if np.array_equal(user_answer, correct_answer):
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect. The correct answer is:")
                        st.write(pd.DataFrame(correct_answer))
                except:
                    st.error("Invalid input. Please enter a valid 2x2 matrix.")

        elif quiz_type == "Matrix Subtraction":
            st.write("**Quiz: Matrix Subtraction**")
            st.write("Subtract the following matrices:")
            A = np.array([[1, 2], [3, 4]])
            B = np.array([[5, 6], [7, 8]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            st.write("B =")
            st.write(pd.DataFrame(B))
            user_answer = input_matrix_table(2, 2, "quiz_sub")
            if st.button("Submit"):
                try:
                    correct_answer = matrix_subtraction(A, B)
                    if np.array_equal(user_answer, correct_answer):
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect. The correct answer is:")
                        st.write(pd.DataFrame(correct_answer))
                except:
                    st.error("Invalid input. Please enter a valid 2x2 matrix.")

        elif quiz_type == "Matrix Multiplication":
            st.write("**Quiz: Matrix Multiplication**")
            st.write("Multiply the following matrices:")
            A = np.array([[1, 2], [3, 4]])
            B = np.array([[5, 6], [7, 8]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            st.write("B =")
            st.write(pd.DataFrame(B))
            user_answer = input_matrix_table(2, 2, "quiz_mul")
            if st.button("Submit"):
                try:
                    correct_answer = matrix_multiplication(A, B)
                    if np.array_equal(user_answer, correct_answer):
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect. The correct answer is:")
                        st.write(pd.DataFrame(correct_answer))
                except:
                    st.error("Invalid input. Please enter a valid 2x2 matrix.")

        elif quiz_type == "Matrix Transpose":
            st.write("**Quiz: Matrix Transpose**")
            st.write("Find the transpose of the following matrix:")
            A = np.array([[1, 2, 3], [4, 5, 6]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            user_answer = input_matrix_table(3, 2, "quiz_trans")
            if st.button("Submit"):
                try:
                    correct_answer = matrix_transpose(A)
                    if np.array_equal(user_answer, correct_answer):
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect. The correct answer is:")
                        st.write(pd.DataFrame(correct_answer))
                except:
                    st.error("Invalid input. Please enter a valid 3x2 matrix.")

        elif quiz_type == "Matrix Determinant":
            st.write("**Quiz: Matrix Determinant**")
            st.write("Find the determinant of the following matrix:")
            A = np.array([[1, 2], [3, 4]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            user_answer = st.number_input("Enter your answer:", value=0, step=1, format="%d")
            if st.button("Submit"):
                correct_answer = matrix_determinant(A)
                if user_answer == correct_answer:
                    st.success("Correct! 🎉")
                else:
                    st.error(f"Incorrect. The correct answer is: {correct_answer}")

        elif quiz_type == "Matrix Inverse":
            st.write("**Quiz: Matrix Inverse**")
            st.write("Find the inverse of the following matrix:")
            A = np.array([[1, 2], [3, 4]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            user_answer = input_matrix_table(2, 2, "quiz_inv")
            if st.button("Submit"):
                try:
                    correct_answer = matrix_inverse(A)
                    if np.array_equal(user_answer, correct_answer):
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect. The correct answer is:")
                        st.write(pd.DataFrame(correct_answer))
                except:
                    st.error("Invalid input. Please enter a valid 2x2 matrix.")

        elif quiz_type == "Scalar Multiplication":
            st.write("**Quiz: Scalar Multiplication**")
            st.write("Multiply the following matrix by the scalar 2:")
            A = np.array([[1, 2], [3, 4]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            user_answer = input_matrix_table(2, 2, "quiz_scalar")
            if st.button("Submit"):
                try:
                    correct_answer = scalar_multiplication(A, 2)
                    if np.array_equal(user_answer, correct_answer):
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect. The correct answer is:")
                        st.write(pd.DataFrame(correct_answer))
                except:
                    st.error("Invalid input. Please enter a valid 2x2 matrix.")

        elif quiz_type == "Matrix Power":
            st.write("**Quiz: Matrix Power**")
            st.write("Raise the following matrix to the power of 2:")
            A = np.array([[1, 2], [3, 4]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            user_answer = input_matrix_table(2, 2, "quiz_power")
            if st.button("Submit"):
                try:
                    correct_answer = matrix_power(A, 2)
                    if np.array_equal(user_answer, correct_answer):
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect. The correct answer is:")
                        st.write(pd.DataFrame(correct_answer))
                except:
                    st.error("Invalid input. Please enter a valid 2x2 matrix.")

        elif quiz_type == "Matrix Rank":
            st.write("**Quiz: Matrix Rank**")
            st.write("Find the rank of the following matrix:")
            A = np.array([[1, 2], [3, 4]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            user_answer = st.number_input("Enter your answer:", value=0, step=1, format="%d")
            if st.button("Submit"):
                correct_answer = matrix_rank(A)
                if user_answer == correct_answer:
                    st.success("Correct! 🎉")
                else:
                    st.error(f"Incorrect. The correct answer is: {correct_answer}")

        elif quiz_type == "Matrix Trace":
            st.write("**Quiz: Matrix Trace**")
            st.write("Find the trace of the following matrix:")
            A = np.array([[1, 2], [3, 4]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            user_answer = st.number_input("Enter your answer:", value=0, step=1, format="%d")
            if st.button("Submit"):
                correct_answer = matrix_trace(A)
                if user_answer == correct_answer:
                    st.success("Correct! 🎉")
                else:
                    st.error(f"Incorrect. The correct answer is: {correct_answer}")

        elif quiz_type == "Matrix Diagonal":
            st.write("**Quiz: Matrix Diagonal**")
            st.write("Find the diagonal of the following matrix:")
            A = np.array([[1, 2], [3, 4]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            user_answer = st.text_input("Enter your answer (comma-separated):", value="0,0")
            if st.button("Submit"):
                try:
                    correct_answer = matrix_diagonal(A)
                    user_answer_list = [int(x.strip()) for x in user_answer.split(",")]
                    if np.array_equal(user_answer_list, correct_answer):
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect. The correct answer is: {correct_answer}")
                except:
                    st.error("Invalid input. Please enter a valid comma-separated list of numbers.")

        elif quiz_type == "Eigenvalues & Eigenvectors":
            st.write("**Quiz: Eigenvalues & Eigenvectors**")
            st.write("Find the eigenvalues and eigenvectors of the following matrix:")
            A = np.array([[1, 2], [3, 4]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            user_eigenvalues = st.text_input("Enter eigenvalues (comma-separated):", value="0,0")
            user_eigenvectors = st.text_input("Enter eigenvectors (comma-separated, row-wise):", value="0,0,0,0")
            if st.button("Submit"):
                try:
                    eigenvalues, eigenvectors = matrix_eigenvalues_and_eigenvectors(A)
                    user_eigenvalues_list = [float(x.strip()) for x in user_eigenvalues.split(",")]
                    user_eigenvectors_list = [float(x.strip()) for x in user_eigenvectors.split(",")]
                    if np.allclose(user_eigenvalues_list, eigenvalues) and np.allclose(user_eigenvectors_list, eigenvectors.flatten()):
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect. The correct eigenvalues are: {eigenvalues}\nThe correct eigenvectors are:")
                        st.write(pd.DataFrame(eigenvectors))
                except:
                    st.error("Invalid input. Please enter valid comma-separated numbers.")

        elif quiz_type == "Solve Linear Equation":
            st.write("**Quiz: Solve Linear Equation**")
            st.write("Solve the following linear equation Ax = b:")
            A = np.array([[1, 2], [3, 4]])
            b = [5, 6]
            st.write("A =")
            st.write(pd.DataFrame(A))
            st.write("b =")
            st.write(b)
            user_answer = st.text_input("Enter your answer (comma-separated):", value="0,0")
            if st.button("Submit"):
                try:
                    correct_answer = matrix_solve_linear_equation(A, b)
                    user_answer_list = [float(x.strip()) for x in user_answer.split(",")]
                    if np.allclose(user_answer_list, correct_answer):
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect. The correct answer is: {correct_answer}")
                except:
                    st.error("Invalid input. Please enter a valid comma-separated list of numbers.")

        elif quiz_type == "Matrix Norm":
            st.write("**Quiz: Matrix Norm**")
            st.write("Find the Frobenius norm of the following matrix:")
            A = np.array([[1, 2], [3, 4]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            user_answer = st.number_input("Enter your answer:", value=0.0, step=0.1, format="%f")
            if st.button("Submit"):
                correct_answer = matrix_norm(A, ord='fro')
                if np.isclose(user_answer, correct_answer):
                    st.success("Correct! 🎉")
                else:
                    st.error(f"Incorrect. The correct answer is: {correct_answer}")

        elif quiz_type == "SVD":
            st.write("**Quiz: Singular Value Decomposition (SVD)**")
            st.write("Find the singular values of the following matrix:")
            A = np.array([[1, 2], [3, 4]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            user_answer = st.text_input("Enter singular values (comma-separated):", value="0,0")
            if st.button("Submit"):
                try:
                    _, S, _ = matrix_svd(A)
                    user_answer_list = [float(x.strip()) for x in user_answer.split(",")]
                    if np.allclose(user_answer_list, S):
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect. The correct singular values are: {S}")
                except:
                    st.error("Invalid input. Please enter valid comma-separated numbers.")

        elif quiz_type == "Cholesky":
            st.write("**Quiz: Cholesky Decomposition**")
            st.write("Find the Cholesky decomposition of the following matrix:")
            A = np.array([[1, 2], [2, 5]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            user_answer = input_matrix_table(2, 2, "quiz_cholesky")
            if st.button("Submit"):
                try:
                    correct_answer = matrix_cholesky_decomposition(A)
                    if np.array_equal(user_answer, correct_answer):
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect. The correct answer is:")
                        st.write(pd.DataFrame(correct_answer))
                except:
                    st.error("Invalid input. Please enter a valid 2x2 matrix.")

        elif quiz_type == "QR":
            st.write("**Quiz: QR Decomposition**")
            st.write("Find the QR decomposition of the following matrix:")
            A = np.array([[1, 2], [3, 4]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            user_answer_Q = input_matrix_table(2, 2, "quiz_qr_Q")
            user_answer_R = input_matrix_table(2, 2, "quiz_qr_R")
            if st.button("Submit"):
                try:
                    Q, R = matrix_qr_decomposition(A)
                    if np.array_equal(user_answer_Q, Q) and np.array_equal(user_answer_R, R):
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect. The correct Q is:")
                        st.write(pd.DataFrame(Q))
                        st.error("The correct R is:")
                        st.write(pd.DataFrame(R))
                except:
                    st.error("Invalid input. Please enter valid 2x2 matrices.")

        elif quiz_type == "LU":
            st.write("**Quiz: LU Decomposition**")
            st.write("Find the LU decomposition of the following matrix:")
            A = np.array([[1, 2], [3, 4]])
            st.write("A =")
            st.write(pd.DataFrame(A))
            user_answer_L = input_matrix_table(2, 2, "quiz_lu_L")
            user_answer_U = input_matrix_table(2, 2, "quiz_lu_U")
            if st.button("Submit"):
                try:
                    P, L, U = scipy_lu(A)  # Use scipy.linalg.lu
                    if np.array_equal(user_answer_L, L) and np.array_equal(user_answer_U, U):
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Incorrect. The correct L is:")
                        st.write(pd.DataFrame(L))
                        st.error("The correct U is:")
                        st.write(pd.DataFrame(U))
                except:
                    st.error("Invalid input. Please enter valid 2x2 matrices.")

# Linear Equation Solver
def linear_equation_solver():
    st.sidebar.header("Linear Equation Solver")
    st.sidebar.write("Solve the linear equation Ax = b")
    
    rows = st.sidebar.number_input("Matrix size:", min_value=1, max_value=4, value=2)
    A = input_matrix_table(rows, rows, "sidebar_matrix")
    
    b_values = []
    st.sidebar.write("Enter b values:")
    for i in range(rows):
        b_values.append(st.sidebar.number_input(f"b[{i+1}]:", value=0, key=f"sidebar_b_{i}"))
    
    if st.sidebar.button("Solve Linear Equation"):
        try:
            solution = matrix_solve_linear_equation(A, np.array(b_values))
            st.sidebar.success("Solution x:")
            solution_df = pd.DataFrame(solution, columns=["Value"], index=[f"x{i+1}" for i in range(len(solution))])
            st.sidebar.write(solution_df)
        except np.linalg.LinAlgError:
            st.sidebar.error("Matrix A is singular or not invertible.")

# Main function
def main():
    load_css()
    st.title("Ze Matrix - Advanced Matrix Operations")
    st.write("A comprehensive tool for learning and practicing matrix operations.")

    # Sidebar for Educational Mode and Linear Equation Solver
    educational_mode()
    linear_equation_solver()

    # Matrix Order Selection
    st.header("Matrix Order Selection")
    st.write("Select the number of rows and columns for the matrix:")

    # Use radio buttons for rows and columns
    rows = st.radio("Rows", [1, 2, 3, 4], key="rows")
    cols = st.radio("Columns", [1, 2, 3, 4], key="cols")

    # Matrix Input
    st.header("Matrix Input")
    matrix = input_matrix_table(rows, cols, "matrix")

    # Add More Matrices
    add_another_matrix = st.checkbox("Add another matrix")
    matrices = [matrix]
    if add_another_matrix:
        rows2 = st.radio("Rows for Matrix 2", [1, 2, 3, 4], key="rows2")
        cols2 = st.radio("Columns for Matrix 2", [1, 2, 3, 4], key="cols2")
        matrix2 = input_matrix_table(rows2, cols2, "matrix2")
        matrices.append(matrix2)

    # Scalar Input (Optional)
    use_scalar = st.checkbox("Use a scalar value")
    scalar = None
    if use_scalar:
        scalar = st.number_input("Enter the scalar value", value=1, step=1, format="%d")

    # Operation Selection
    st.header("Operation Selection")
    operation = st.selectbox("Select Operation", [
        "Matrix Addition", "Matrix Subtraction", "Scalar Multiplication",
        "Matrix Multiplication", "Matrix Transpose", "Matrix Power",
        "Matrix Determinant", "Matrix Inverse", "Matrix Rank",
        "Matrix Trace", "Matrix Diagonal", "Eigenvalues & Eigenvectors", "Matrix Norm", "SVD", "Cholesky", "QR", "LU"
    ])

    # Perform Operation
    if st.button("Perform Operation"):
        if operation == "Matrix Addition":
            if len(matrices) == 2:
                result = matrix_addition(matrices[0], matrices[1])
            else:
                st.error("Matrix Addition requires exactly two matrices.")
                return
        elif operation == "Matrix Subtraction":
            if len(matrices) == 2:
                result = matrix_subtraction(matrices[0], matrices[1])
            else:
                st.error("Matrix Subtraction requires exactly two matrices.")
                return
        elif operation == "Scalar Multiplication":
            if scalar is not None:
                result = scalar_multiplication(matrices[0], scalar)
            else:
                st.error("Scalar value is required for Scalar Multiplication.")
                return
        elif operation == "Matrix Multiplication":
            if len(matrices) == 2:
                if matrices[0].shape[1] == matrices[1].shape[0]:
                    result = matrix_multiplication(matrices[0], matrices[1])
                else:
                    st.error("Number of columns in Matrix 1 must equal number of rows in Matrix 2 for multiplication.")
                    return
            else:
                st.error("Matrix Multiplication requires exactly two matrices.")
                return
        elif operation == "Matrix Transpose":
            result = matrix_transpose(matrices[0])
        elif operation == "Matrix Power":
            if matrices[0].shape[0] == matrices[0].shape[1]:
                power = st.sidebar.number_input("Enter the power", min_value=1, value=2, step=1, format="%d")
                result = matrix_power(matrices[0], power)
            else:
                st.error("Matrix must be square to calculate power.")
                return
        elif operation == "Matrix Determinant":
            if matrices[0].shape[0] == matrices[0].shape[1]:
                result = matrix_determinant(matrices[0])
                st.write(f"The Determinant of the matrix is {result}")
                return
            else:
                st.error("Matrix must be square to calculate determinant.")
                return
        elif operation == "Matrix Inverse":
            if matrices[0].shape[0] == matrices[0].shape[1]:
                try:
                    result = matrix_inverse(matrices[0])
                except np.linalg.LinAlgError:
                    st.error("Matrix is singular and does not have an inverse.")
                    return
            else:
                st.error("Matrix must be square to calculate inverse.")
                return
        elif operation == "Matrix Rank":
            result = matrix_rank(matrices[0])
            st.write(f"Rank of the matrix is {result}")
            return
        elif operation == "Matrix Trace":
            if matrices[0].shape[0] == matrices[0].shape[1]:
                result = matrix_trace(matrices[0])
                st.write(f"Trace of the matrix is {result}")
                return
            else:
                st.error("Matrix must be square to calculate trace.")
                return
        elif operation == "Matrix Diagonal":
            if matrices[0].shape[0] == matrices[0].shape[1]:
                result = matrix_diagonal(matrices[0])
            else:
                st.error("Matrix must be square to extract diagonal.")
                return
        elif operation == "Eigenvalues & Eigenvectors":
            if matrices[0].shape[0] == matrices[0].shape[1]:
                eigenvalues, eigenvectors = matrix_eigenvalues_and_eigenvectors(matrices[0])
                st.write("Eigenvalues:")
                st.write(eigenvalues)
                st.write("Eigenvectors:")
                st.write(pd.DataFrame(eigenvectors))
                return
            else:
                st.error("Matrix must be square to calculate eigenvalues and eigenvectors.")
                return
        
        
        elif operation == "Matrix Norm":
            ord = st.sidebar.selectbox("Select norm type", [None, "fro", "nuc", np.inf, -np.inf, 1, -1, 2, -2], key="norm_type")
            result = matrix_norm(matrices[0], ord)
            st.write(f"The Norm of the matrix is {result}")
            return
        elif operation == "SVD":
            U, S, V = matrix_svd(matrices[0])
            st.write("U (Left singular vectors):")
            st.write(pd.DataFrame(U))
            st.write("S (Singular values):")
            st.write(S)
            st.write("V (Right singular vectors):")
            st.write(pd.DataFrame(V))
            return
        elif operation == "Cholesky":
            if matrices[0].shape[0] == matrices[0].shape[1]:
                try:
                    result = matrix_cholesky_decomposition(matrices[0])
                except np.linalg.LinAlgError:
                    st.error("Matrix is not positive definite.")
                    return
            else:
                st.error("Matrix must be square for Cholesky decomposition.")
                return
        elif operation == "QR":
            Q, R = matrix_qr_decomposition(matrices[0])
            st.write("Q (Orthogonal matrix):")
            st.write(pd.DataFrame(Q))
            st.write("R (Upper triangular matrix):")
            st.write(pd.DataFrame(R))
            return
        elif operation == "LU":
            P, L, U = scipy_lu(matrices[0])  # Use scipy.linalg.lu
            st.write("P (Permutation matrix):")
            st.write(pd.DataFrame(P))
            st.write("L (Lower triangular matrix):")
            st.write(pd.DataFrame(L))
            st.write("U (Upper triangular matrix):")
            st.write(pd.DataFrame(U))
            return

        st.write("Result of the Operation:")
        st.write(pd.DataFrame(result))

if __name__ == "__main__":
    main()
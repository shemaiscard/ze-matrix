# app.py
import json
import random

import streamlit as st
import numpy as np
import pandas as pd
from scipy.linalg import lu as scipy_lu

from linalg_steps import (det_steps, fmt, inverse_steps, lu_steps, parse_entry, qr_steps,
                          rank_steps, rref, solve_steps, svd_steps, to_float)
from quizzes import QUIZ_TYPES, check, new_question

SIZES = [1, 2, 3, 4, 5, 6, 7, 8]

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

# Matrix input: an editable grid that accepts integers, decimals and fractions
def matrix_editor(rows, cols, key, label=None, where=st):
    if label:
        where.write(f"{label} =")
    df = pd.DataFrame([["0"] * cols for _ in range(rows)],
                      columns=[str(j + 1) for j in range(cols)], index=[str(i + 1) for i in range(rows)])
    edited = where.data_editor(df, key=f"{key}_{rows}x{cols}", use_container_width=True)
    values, bad = [], []
    for i in range(rows):
        row = []
        for j in range(cols):
            try:
                row.append(parse_entry(edited.iat[i, j]))
            except (ValueError, ZeroDivisionError):
                bad.append(f"[{i + 1}][{j + 1}]")
        values.append(row)
    if bad:
        where.error(f"Not a number in {label or 'the matrix'}: {', '.join(bad)}. Use forms like 3, -0.5 or 1/3.")
        return None
    return values

def parse_list(text):
    return [parse_entry(x) for x in str(text).replace(";", ",").split(",") if x.strip()]

def as_frame(matrix, sep=None):
    cols = len(matrix[0]) if matrix else 0
    names = [str(j + 1) if sep is None or j < sep else f"| {j - sep + 1}" for j in range(cols)]
    return pd.DataFrame([[fmt(x) for x in row] for row in matrix], columns=names,
                        index=[str(i + 1) for i in range(len(matrix))])

def show_value(label, value, where=st):
    if isinstance(value, (list, np.ndarray)) and len(value) and isinstance(value[0], (list, np.ndarray)):
        where.write(f"{label} =")
        where.dataframe(as_frame([list(r) for r in value]))
    elif isinstance(value, (list, np.ndarray)):
        where.write(f"{label} = [ " + ",  ".join(fmt(x) for x in value) + " ]")
    else:
        where.write(f"{label} = {fmt(value)}")

def show_steps(steps, where=st):
    with where.expander("Show the steps"):
        for i, s in enumerate(steps, 1):
            st.markdown(f"**{i}.** {s['text']}")
            if s["matrix"] is not None:
                st.dataframe(as_frame(s["matrix"], s["sep"]))

def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

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
        quiz_type = st.selectbox("Select a Quiz Type", QUIZ_TYPES)
        if "quiz_rng" not in st.session_state:
            st.session_state.quiz_rng = random.Random()
            st.session_state.quiz_questions = {}
            st.session_state.quiz_round = 0
            st.session_state.progress = {}
        questions = st.session_state.quiz_questions
        new = st.button("New question", key="quiz_new")
        if new or quiz_type not in questions:
            questions[quiz_type] = new_question(quiz_type, st.session_state.quiz_rng)
            st.session_state.quiz_round += 1  # fresh, empty answer boxes
        q = questions[quiz_type]

        st.write(f"**Quiz: {quiz_type}**")
        st.write(q["prompt"])
        for label, value in q["given"]:
            show_value(label, value)

        answers, bad = {}, []
        for label, kind, shape in q["fields"]:
            key = f"quiz_{quiz_type}_{label}_{st.session_state.quiz_round}"
            if kind == "matrix":
                value = matrix_editor(*shape, key, label=label)
                if value is None:
                    bad.append(label)
            else:
                text = st.text_input(f"{label}" + (" (comma-separated)" if kind == "vector" else ""), key=key)
                try:
                    value = parse_list(text) if kind == "vector" else parse_entry(text)
                    if kind == "vector" and len(value) != shape:
                        bad.append(label)
                except (ValueError, ZeroDivisionError):
                    bad.append(label)
            answers[label] = value

        if st.button("Submit", key="quiz_submit"):
            if bad:
                st.error("Please complete: " + ", ".join(bad))
            else:
                correct, solution = check(q, answers)
                score = st.session_state.progress.setdefault(quiz_type, {"attempts": 0, "correct": 0})
                score["attempts"] += 1
                score["correct"] += int(correct)
                if correct:
                    st.success("Correct! Press New question for another one.")
                else:
                    st.error("Incorrect. The correct answer is:")
                    for label, value in solution:
                        show_value(label, value)

        progress = st.session_state.progress
        if quiz_type in progress:
            s = progress[quiz_type]
            st.caption(f"Your score in this topic: {s['correct']} of {s['attempts']}")
        with st.expander("Your progress"):
            if progress:
                st.dataframe(pd.DataFrame([{"Topic": k, "Correct": v["correct"], "Attempts": v["attempts"]}
                                           for k, v in progress.items()]), hide_index=True)
            else:
                st.write("No answers yet.")
            st.download_button("Download progress", json.dumps(progress, indent=1),
                               file_name="ze_matrix_progress.json", mime="application/json")
            uploaded = st.file_uploader("Continue from a saved progress file", type="json")
            if uploaded is not None and st.session_state.get("progress_file") != uploaded.file_id:
                try:
                    loaded = json.load(uploaded)
                    st.session_state.progress = {k: {"attempts": int(v["attempts"]), "correct": int(v["correct"])}
                                                 for k, v in loaded.items() if k in QUIZ_TYPES}
                    st.session_state.progress_file = uploaded.file_id
                    st.rerun()
                except (ValueError, KeyError, TypeError, AttributeError):
                    st.error("This is not a Ze Matrix progress file.")

# Linear Equation Solver
def linear_equation_solver():
    st.sidebar.header("Linear Equation Solver")
    st.sidebar.write("Solve the linear equation Ax = b")

    # Input matrix size
    rows = st.sidebar.selectbox("Matrix size:", SIZES[:6], index=1)

    # Input matrix A and vector b in the sidebar
    A = matrix_editor(rows, rows, "sidebar_matrix", label="A", where=st.sidebar)
    b = matrix_editor(rows, 1, "sidebar_b", label="b", where=st.sidebar)

    # Solve the linear equation by row reduction of [A | b]
    if st.sidebar.button("Solve Linear Equation") and A is not None and b is not None:
        kind, x, steps = solve_steps(A, [row[0] for row in b])
        if kind == "unique":
            st.sidebar.success("Solution x:")
            st.sidebar.dataframe(pd.DataFrame([fmt(v) for v in x], columns=["Value"],
                                              index=[f"x{i + 1}" for i in range(len(x))]))
        elif kind == "none":
            st.sidebar.error("The system has no solution.")
        else:
            st.sidebar.warning("The system has infinitely many solutions.")
        show_steps(steps, where=st.sidebar)

# Main function
def main():
    st.set_page_config(page_title="Ze Matrix")
    load_css()
    st.title("Ze Matrix - Advanced Matrix Operations")
    st.write("A tool for learning and practicing matrix operations.")

    # Sidebar for Educational Mode and Linear Equation Solver
    educational_mode()
    linear_equation_solver()

    # Matrix Order Selection
    st.header("Matrix Order Selection")
    st.write("Select the number of rows and columns for the matrix:")
    rows = st.radio("Rows", SIZES, key="rows", horizontal=True)
    cols = st.radio("Columns", SIZES, key="cols", horizontal=True)

    # Matrix Input
    st.header("Matrix Input")
    st.caption("Type integers, decimals or fractions, such as 3, -0.5 or 1/3.")
    A = matrix_editor(rows, cols, "matrix")

    # Add More Matrices
    B = None
    add_another_matrix = st.checkbox("Add another matrix")
    if add_another_matrix:
        rows2 = st.radio("Rows for Matrix 2", SIZES, key="rows2", horizontal=True)
        cols2 = st.radio("Columns for Matrix 2", SIZES, key="cols2", horizontal=True)
        B = matrix_editor(rows2, cols2, "matrix2")

    # Scalar Input (Optional)
    scalar = None
    if st.checkbox("Use a scalar value"):
        try:
            scalar = parse_entry(st.text_input("Enter the scalar value", value="1"))
        except (ValueError, ZeroDivisionError):
            st.error("The scalar must be a number such as 2, -0.5 or 1/3.")

    # Operation Selection
    st.header("Operation Selection")
    operation = st.selectbox("Select Operation", [
        "Matrix Addition", "Matrix Subtraction", "Scalar Multiplication",
        "Matrix Multiplication", "Matrix Transpose", "Matrix Power",
        "Matrix Determinant", "Matrix Inverse", "Matrix Rank", "Row Reduction (RREF)",
        "Matrix Trace", "Matrix Diagonal", "Eigenvalues & Eigenvectors", "Matrix Norm", "SVD", "Cholesky", "QR", "LU"
    ])

    # Options needed by some operations (chosen before pressing the button)
    if operation == "Matrix Power":
        power = st.number_input("Enter the power", min_value=0, max_value=50, value=2, step=1)
    if operation == "Matrix Norm":
        norm_labels = {"Frobenius": "fro", "Nuclear": "nuc", "Max row sum (inf)": np.inf,
                       "Max column sum (1)": 1, "Spectral (2)": 2}
        norm_choice = st.selectbox("Select norm type", list(norm_labels))

    # Perform Operation
    if not st.button("Perform Operation"):
        return
    if A is None or (add_another_matrix and B is None):
        st.error("Please fix the matrix entries first.")
        return
    n_rows, n_cols = len(A), len(A[0])
    square = n_rows == n_cols
    Af = to_float(A)

    if operation in ("Matrix Addition", "Matrix Subtraction", "Matrix Multiplication"):
        if B is None:
            st.error(f"{operation} requires exactly two matrices.")
            return
        if operation == "Matrix Multiplication":
            if n_cols != len(B):
                st.error("Number of columns in Matrix 1 must equal number of rows in Matrix 2 for multiplication.")
                return
            result = matmul(A, B)
        else:
            if (n_rows, n_cols) != (len(B), len(B[0])):
                st.error("Both matrices must have the same size.")
                return
            sign = 1 if operation == "Matrix Addition" else -1
            result = [[a + sign * b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]
    elif operation == "Scalar Multiplication":
        if scalar is None:
            st.error("Scalar value is required for Scalar Multiplication.")
            return
        result = [[scalar * x for x in row] for row in A]
    elif operation == "Matrix Transpose":
        result = [list(col) for col in zip(*A)]
    elif operation in ("Matrix Power", "Matrix Determinant", "Matrix Inverse", "Matrix Trace",
                       "Matrix Diagonal", "Eigenvalues & Eigenvectors", "Cholesky") and not square:
        st.error(f"Matrix must be square for {operation}.")
        return
    elif operation == "Matrix Power":
        result = [[type(A[0][0])(int(i == j)) for j in range(n_cols)] for i in range(n_rows)]
        for _ in range(int(power)):
            result = matmul(result, A)
    elif operation == "Matrix Determinant":
        det, steps = det_steps(A)
        st.write(f"The Determinant of the matrix is {fmt(det)}")
        show_steps(steps)
        return
    elif operation == "Matrix Inverse":
        result, steps = inverse_steps(A)
        if result is None:
            st.error("Matrix is singular and does not have an inverse.")
            show_steps(steps)
            return
        st.write("Result of the Operation:")
        st.dataframe(as_frame(result))
        show_steps(steps)
        return
    elif operation == "Matrix Rank":
        rank, _, steps = rank_steps(A)
        st.write(f"Rank of the matrix is {rank}")
        show_steps(steps)
        return
    elif operation == "Row Reduction (RREF)":
        steps = []
        result, _ = rref(A, steps=steps)
        st.write("Reduced row echelon form:")
        st.dataframe(as_frame(result))
        show_steps(steps)
        return
    elif operation == "Matrix Trace":
        st.write(f"Trace of the matrix is {fmt(sum(A[i][i] for i in range(n_rows)))}")
        return
    elif operation == "Matrix Diagonal":
        result = [[A[i][i] for i in range(n_rows)]]
    elif operation == "Eigenvalues & Eigenvectors":
        eigenvalues, eigenvectors = matrix_eigenvalues_and_eigenvectors(Af)
        st.write("Eigenvalues:")
        st.write("[ " + ",  ".join(fmt(x) for x in eigenvalues) + " ]")
        st.write("Eigenvectors (columns, scaled to length 1):")
        st.dataframe(as_frame(eigenvectors.tolist()))
        return
    elif operation == "Matrix Norm":
        st.write(f"The {norm_choice} norm of the matrix is {fmt(matrix_norm(Af, norm_labels[norm_choice]))}")
        return
    elif operation == "SVD":
        U, S, V = matrix_svd(Af)
        st.write("U (Left singular vectors):")
        st.dataframe(as_frame(U.tolist()))
        st.write("S (Singular values): [ " + ",  ".join(fmt(x) for x in S) + " ]")
        st.write("V^T (Right singular vectors, as rows):")
        st.dataframe(as_frame(V.tolist()))
        show_steps(svd_steps(Af)[2])
        return
    elif operation == "Cholesky":
        if not np.allclose(Af, Af.T):
            st.error("Cholesky needs a symmetric matrix.")
            return
        try:
            result = matrix_cholesky_decomposition(Af).tolist()
        except np.linalg.LinAlgError:
            st.error("Matrix is not positive definite.")
            return
    elif operation == "QR":
        Q, R = matrix_qr_decomposition(Af)
        st.write("Q (Orthogonal matrix):")
        st.dataframe(as_frame(Q.tolist()))
        st.write("R (Upper triangular matrix):")
        st.dataframe(as_frame(R.tolist()))
        if n_rows >= n_cols:
            Qg, Rg, steps = qr_steps(Af)
            if Qg is not None:
                steps.append({"text": "Gram-Schmidt result (it can differ from the table above by the sign of a column of Q and the matching row of R)",
                              "matrix": None, "sep": None})
            show_steps(steps)
        return
    elif operation == "LU":
        if square:
            P, L, U, steps = lu_steps(A)
            st.write("P (Permutation matrix), with P A = L U:")
        else:
            P, L, U = scipy_lu(Af)
            P, steps = P.T, None
            st.write("P (Permutation matrix), with P A = L U:")
        st.dataframe(as_frame(P.tolist() if hasattr(P, "tolist") else P))
        st.write("L (Lower triangular matrix):")
        st.dataframe(as_frame(L.tolist() if hasattr(L, "tolist") else L))
        st.write("U (Upper triangular matrix):")
        st.dataframe(as_frame(U.tolist() if hasattr(U, "tolist") else U))
        if steps:
            show_steps(steps)
        return

    st.write("Result of the Operation:")
    st.dataframe(as_frame(result))

if __name__ == "__main__":
    main()

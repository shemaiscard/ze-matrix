# quizzes.py
# Randomly generated quiz questions. Matrices are built so that answers stay
# readable: integer or simple-fraction inverses, integer eigenvalues
# (A = P D P^-1 with det P = 1), Cholesky factors with integer entries, and
# QR problems whose first column has a whole-number length.
from fractions import Fraction as F

import numpy as np

from linalg_steps import det_steps, fmt, inverse_steps, rank_steps

QUIZ_TYPES = [
    "Matrix Addition", "Matrix Subtraction", "Matrix Multiplication",
    "Matrix Transpose", "Matrix Determinant", "Matrix Inverse",
    "Scalar Multiplication", "Matrix Power", "Matrix Rank",
    "Matrix Trace", "Matrix Diagonal", "Eigenvalues & Eigenvectors",
    "Solve Linear Equation", "Matrix Norm", "SVD", "Cholesky", "QR", "LU",
]

TOL = 0.01  # for answers that need square roots


def M(rows):
    return [[F(x) for x in row] for row in rows]


def rand_matrix(rng, r, c, lo=-5, hi=5):
    return M([[rng.randint(lo, hi) for _ in range(c)] for _ in range(r)])


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def transpose(A):
    return [list(col) for col in zip(*A)]


def nonsingular(rng, n, lo=-4, hi=4):
    while True:
        A = rand_matrix(rng, n, n, lo, hi)
        if det_steps(A)[0] != 0:
            return A


def new_question(qtype, rng):
    """Returns a dict: prompt, given [(label, matrix or list)], fields [(label, kind, shape)], data."""
    q = {"type": qtype, "given": [], "fields": [], "data": {}}
    if qtype in ("Matrix Addition", "Matrix Subtraction"):
        A, B = rand_matrix(rng, 2, 2, -9, 9), rand_matrix(rng, 2, 2, -9, 9)
        op = "+" if qtype == "Matrix Addition" else "-"
        q["prompt"] = f"Compute A {op} B."
        q["given"] = [("A", A), ("B", B)]
        q["fields"] = [(f"A {op} B", "matrix", (2, 2))]
        q["data"]["answer"] = [[a + b if op == "+" else a - b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]
    elif qtype == "Matrix Multiplication":
        A, B = rand_matrix(rng, 2, 2, -4, 4), rand_matrix(rng, 2, 2, -4, 4)
        q["prompt"] = "Compute the product AB."
        q["given"] = [("A", A), ("B", B)]
        q["fields"] = [("AB", "matrix", (2, 2))]
        q["data"]["answer"] = matmul(A, B)
    elif qtype == "Matrix Transpose":
        A = rand_matrix(rng, 2, 3, -9, 9)
        q["prompt"] = "Write the transpose of A."
        q["given"] = [("A", A)]
        q["fields"] = [("A^T", "matrix", (3, 2))]
        q["data"]["answer"] = transpose(A)
    elif qtype == "Matrix Determinant":
        n = rng.choice([2, 3])
        A = rand_matrix(rng, n, n, -4, 4)
        q["prompt"] = "Find det(A)."
        q["given"] = [("A", A)]
        q["fields"] = [("det(A)", "scalar", None)]
        q["data"]["answer"] = det_steps(A)[0]
    elif qtype == "Matrix Inverse":
        A = nonsingular(rng, 2)
        q["prompt"] = "Find the inverse of A. Fractions such as 1/3 are accepted."
        q["given"] = [("A", A)]
        q["fields"] = [("A^-1", "matrix", (2, 2))]
        q["data"]["answer"] = inverse_steps(A)[0]
    elif qtype == "Scalar Multiplication":
        k = rng.choice([-4, -3, -2, 2, 3, 4, 5])
        A = rand_matrix(rng, 2, 2, -6, 6)
        q["prompt"] = f"Compute {k}A."
        q["given"] = [("A", A)]
        q["fields"] = [(f"{k}A", "matrix", (2, 2))]
        q["data"]["answer"] = [[k * x for x in row] for row in A]
    elif qtype == "Matrix Power":
        p = rng.choice([2, 3])
        A = rand_matrix(rng, 2, 2, -2, 2)
        q["prompt"] = f"Compute A^{p}."
        q["given"] = [("A", A)]
        q["fields"] = [(f"A^{p}", "matrix", (2, 2))]
        R = A
        for _ in range(p - 1):
            R = matmul(R, A)
        q["data"]["answer"] = R
    elif qtype == "Matrix Rank":
        r = rng.choice([1, 2, 3])
        A = matmul(rand_matrix(rng, 3, r, -2, 2), rand_matrix(rng, r, 3, -2, 2))
        q["prompt"] = "Find the rank of A."
        q["given"] = [("A", A)]
        q["fields"] = [("rank(A)", "scalar", None)]
        q["data"]["answer"] = F(rank_steps(A)[0])
    elif qtype == "Matrix Trace":
        A = rand_matrix(rng, 3, 3, -9, 9)
        q["prompt"] = "Find the trace of A."
        q["given"] = [("A", A)]
        q["fields"] = [("tr(A)", "scalar", None)]
        q["data"]["answer"] = sum(A[i][i] for i in range(3))
    elif qtype == "Matrix Diagonal":
        A = rand_matrix(rng, 3, 3, -9, 9)
        q["prompt"] = "List the main diagonal of A, top to bottom."
        q["given"] = [("A", A)]
        q["fields"] = [("diagonal", "vector", 3)]
        q["data"]["answer"] = [A[i][i] for i in range(3)]
    elif qtype == "Eigenvalues & Eigenvectors":
        l1, l2 = rng.sample(range(-4, 6), 2)
        a, b = rng.randint(-2, 2), rng.randint(-2, 2)
        P = M([[1 + a * b, a], [b, 1]])                # det P = 1, so P^-1 has integer entries
        Pinv = M([[1, -a], [-b, 1 + a * b]])
        A = matmul(matmul(P, M([[l1, 0], [0, l2]])), Pinv)
        q["prompt"] = "Find both eigenvalues of A (any order) and one eigenvector for the larger eigenvalue."
        q["given"] = [("A", A)]
        q["fields"] = [("eigenvalues", "vector", 2), ("eigenvector for the larger eigenvalue", "vector", 2)]
        big = max(l1, l2)
        q["data"] = {"A": A, "eigenvalues": sorted([F(l1), F(l2)]), "big": F(big),
                     "vector": [P[0][0 if big == l1 else 1], P[1][0 if big == l1 else 1]]}
    elif qtype == "Solve Linear Equation":
        A = nonsingular(rng, 3, -3, 3)
        x = [F(rng.randint(-4, 4)) for _ in range(3)]
        b = [sum(A[i][j] * x[j] for j in range(3)) for i in range(3)]
        q["prompt"] = "Solve Ax = b."
        q["given"] = [("A", A), ("b", b)]
        q["fields"] = [("x", "vector", 3)]
        q["data"]["answer"] = x
    elif qtype == "Matrix Norm":
        A = rand_matrix(rng, 2, 2, -5, 5)
        q["prompt"] = "Find the Frobenius norm of A (two decimals are enough)."
        q["given"] = [("A", A)]
        q["fields"] = [("||A||_F", "scalar", None)]
        q["data"]["answer"] = float(np.linalg.norm(np.array(A, dtype=float)))
    elif qtype == "SVD":
        A = rand_matrix(rng, 2, 2, -4, 4)
        while all(x == 0 for row in A for x in row):
            A = rand_matrix(rng, 2, 2, -4, 4)
        q["prompt"] = "Find the singular values of A (two decimals are enough)."
        q["given"] = [("A", A)]
        q["fields"] = [("singular values", "vector", 2)]
        q["data"]["answer"] = sorted(np.linalg.svd(np.array(A, dtype=float), compute_uv=False).tolist(), reverse=True)
    elif qtype == "Cholesky":
        L = M([[rng.randint(1, 3), 0], [rng.randint(-3, 3), rng.randint(1, 3)]])
        A = matmul(L, transpose(L))
        q["prompt"] = "Find the lower triangular L with positive diagonal such that A = L L^T."
        q["given"] = [("A", A)]
        q["fields"] = [("L", "matrix", (2, 2))]
        q["data"]["answer"] = L
    elif qtype == "QR":
        x, y = rng.choice([(3, 4), (4, 3), (6, 8), (8, 6), (5, 12), (12, 5)])
        x, y = x * rng.choice([-1, 1]), y * rng.choice([-1, 1])
        while True:
            c2 = [rng.randint(-6, 6), rng.randint(-6, 6)]
            if x * c2[1] - y * c2[0] != 0:
                break
        A = M([[x, c2[0]], [y, c2[1]]])
        q["prompt"] = "Find a QR decomposition of A: Q orthogonal, R upper triangular, A = QR. Decimals or fractions are accepted."
        q["given"] = [("A", A)]
        q["fields"] = [("Q", "matrix", (2, 2)), ("R", "matrix", (2, 2))]
        q["data"]["A"] = A
    elif qtype == "LU":
        n = rng.choice([2, 3])
        L = [[F(1) if i == j else (F(rng.randint(-3, 3)) if j < i else F(0)) for j in range(n)] for i in range(n)]
        U = [[F(rng.choice([-3, -2, -1, 1, 2, 3])) if i == j else (F(rng.randint(-4, 4)) if j > i else F(0))
              for j in range(n)] for i in range(n)]
        A = matmul(L, U)
        q["prompt"] = "Find A = LU without row swaps, with 1s on the diagonal of L (Doolittle form)."
        q["given"] = [("A", A)]
        q["fields"] = [("L", "matrix", (n, n)), ("U", "matrix", (n, n))]
        q["data"] = {"L": L, "U": U}
    else:
        raise ValueError(qtype)
    return q


def _close(a, b, tol=TOL):
    return np.allclose(np.array(a, dtype=float), np.array(b, dtype=float), atol=tol)


def check(q, answers):
    """answers: label -> parsed value (Fraction matrix, list or scalar).
    Returns (correct, solution) where solution is a list of (label, value) to show."""
    t, d = q["type"], q["data"]
    label = q["fields"][0][0]
    if t == "Eigenvalues & Eigenvectors":
        vals, vec = answers["eigenvalues"], answers["eigenvector for the larger eigenvalue"]
        A, lam = d["A"], d["big"]
        ok_vals = sorted(vals) == d["eigenvalues"]
        ok_vec = any(v != 0 for v in vec) and all(
            sum(A[i][j] * vec[j] for j in range(2)) == lam * vec[i] for i in range(2))
        return ok_vals and ok_vec, [("eigenvalues", d["eigenvalues"]),
                                    (f"an eigenvector for {fmt(lam)} (any nonzero multiple is correct)", d["vector"])]
    if t == "QR":
        Q, R = np.array(answers["Q"], dtype=float), np.array(answers["R"], dtype=float)
        A = np.array(d["A"], dtype=float)
        ok = _close(Q.T @ Q, np.eye(2)) and abs(R[1, 0]) < TOL and _close(Q @ R, A, 0.05)
        q_ref, r_ref = np.linalg.qr(A)
        s = np.sign(np.diag(r_ref))                    # show the version with a positive diagonal in R
        return ok, [("Q", (q_ref * s).tolist()), ("R", (r_ref * s[:, None]).tolist())]
    if t == "LU":
        return answers["L"] == d["L"] and answers["U"] == d["U"], [("L", d["L"]), ("U", d["U"])]
    ans = d["answer"]
    given = answers[label]
    if t in ("Matrix Norm", "SVD"):
        if t == "SVD":
            given = sorted((float(x) for x in given), reverse=True)
        return _close(given, ans), [(label, ans)]
    return given == ans, [(label, ans)]

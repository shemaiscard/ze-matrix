# linalg_steps.py
# Step-by-step versions of the main algorithms. Row operations use exact
# fractions, so a student sees 1/3 instead of 0.3333333; QR and SVD need
# square roots and use floats.
from fractions import Fraction

import numpy as np


# ---------------------------------------------------------------- helpers
def parse_entry(text):
    """'3', '-1/2', '0.25' or '' (zero) -> Fraction. Raises ValueError otherwise."""
    text = str(text).strip()
    if text in ("", "-", "+"):
        return Fraction(0)
    return Fraction(text)


def fmt(x):
    """Fraction -> '2', '-1/3'; float -> rounded string."""
    if isinstance(x, Fraction):
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    if isinstance(x, complex) or np.iscomplexobj(x):
        x = complex(x)
        if abs(x.imag) < 1e-12:
            return fmt(x.real)
        return f"{x.real:.4g}{x.imag:+.4g}i"
    x = float(x)
    return "0" if abs(x) < 1e-12 else f"{x:.6g}"


def to_fractions(matrix):
    return [[x if isinstance(x, Fraction) else Fraction(x).limit_denominator(10**9) for x in row] for row in matrix]


def to_float(matrix):
    return np.array([[float(x) for x in row] for row in matrix], dtype=float)


def copy(m):
    return [row[:] for row in m]


def identity(n):
    return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]


def step(steps, text, matrix, sep=None):
    steps.append({"text": text, "matrix": copy(matrix), "sep": sep})


def row_name(i):
    return f"R{i + 1}"


# ---------------------------------------------------------------- Gauss-Jordan
def rref(M, sep=None, steps=None, stop_col=None):
    """Reduced row echelon form with recorded row operations.
    stop_col limits pivot search (for augmented matrices). Returns (R, pivot_cols)."""
    M = to_fractions(M)
    steps = [] if steps is None else steps
    rows, cols = len(M), len(M[0])
    stop_col = cols if stop_col is None else stop_col
    pivots, r = [], 0
    step(steps, "Start", M, sep)
    for c in range(stop_col):
        if r >= rows:
            break
        p = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if p is None:
            continue
        if p != r:
            M[r], M[p] = M[p], M[r]
            step(steps, f"Swap {row_name(r)} and {row_name(p)} to get a nonzero pivot in column {c + 1}", M, sep)
        if M[r][c] != 1:
            k = M[r][c]
            M[r] = [x / k for x in M[r]]
            step(steps, f"{row_name(r)} -> {row_name(r)} / ({fmt(k)}) to make the pivot 1", M, sep)
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
                step(steps, f"{row_name(i)} -> {row_name(i)} - ({fmt(f)}) * {row_name(r)}", M, sep)
        pivots.append(c)
        r += 1
    return M, pivots


def rank_steps(A):
    steps = []
    R, pivots = rref(A, steps=steps)
    steps.append({"text": f"{len(pivots)} pivot(s), in column(s) {', '.join(str(c + 1) for c in pivots) or 'none'}: "
                          f"the rank is {len(pivots)}.", "matrix": None, "sep": None})
    return len(pivots), R, steps


def inverse_steps(A):
    """Gauss-Jordan on [A | I]. Returns (inverse or None, steps)."""
    n = len(A)
    A = to_fractions(A)
    aug = [A[i] + identity(n)[i] for i in range(n)]
    steps = []
    R, pivots = rref(aug, sep=n, steps=steps, stop_col=n)
    if len(pivots) < n:
        steps.append({"text": "The left block cannot become the identity: A is singular and has no inverse.",
                      "matrix": None, "sep": None})
        return None, steps
    steps.append({"text": "The left block is now I, so the right block is the inverse.", "matrix": None, "sep": None})
    return [row[n:] for row in R], steps


def solve_steps(A, b):
    """Row-reduce [A | b]. Returns (kind, x, steps): kind is 'unique', 'none' or 'infinite'."""
    A = to_fractions(A)
    b = [x if isinstance(x, Fraction) else Fraction(x) for x in b]
    n = len(A[0])
    aug = [A[i] + [b[i]] for i in range(len(A))]
    steps = []
    R, pivots = rref(aug, sep=n, steps=steps, stop_col=n)
    if any(all(x == 0 for x in row[:n]) and row[n] != 0 for row in R):
        steps.append({"text": "A row reads 0 = nonzero: the system has no solution.", "matrix": None, "sep": None})
        return "none", None, steps
    if len(pivots) < n:
        free = [c + 1 for c in range(n) if c not in pivots]
        steps.append({"text": f"Free variable(s) x{', x'.join(map(str, free))}: infinitely many solutions.",
                      "matrix": None, "sep": None})
        return "infinite", None, steps
    x = [R[i][n] for i in range(n)]
    steps.append({"text": "Each row now gives one variable directly.", "matrix": None, "sep": None})
    return "unique", x, steps


def det_steps(A):
    """Reduce to upper triangular form; det = (-1)^swaps * product of the diagonal."""
    M = to_fractions(A)
    n = len(M)
    steps, sign = [], 1
    step(steps, "Start", M)
    for c in range(n):
        p = next((i for i in range(c, n) if M[i][c] != 0), None)
        if p is None:
            steps.append({"text": f"Column {c + 1} has no nonzero pivot: the determinant is 0.", "matrix": None, "sep": None})
            return Fraction(0), steps
        if p != c:
            M[c], M[p] = M[p], M[c]
            sign = -sign
            step(steps, f"Swap {row_name(c)} and {row_name(p)} (a swap changes the sign)", M)
        for i in range(c + 1, n):
            if M[i][c] != 0:
                f = M[i][c] / M[c][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[c])]
                step(steps, f"{row_name(i)} -> {row_name(i)} - ({fmt(f)}) * {row_name(c)} (does not change the determinant)", M)
    det = Fraction(sign)
    for i in range(n):
        det *= M[i][i]
    diag = " * ".join(f"({fmt(M[i][i])})" for i in range(n))
    steps.append({"text": f"Upper triangular: det = {'-' if sign < 0 else ''}{diag} = {fmt(det)}", "matrix": None, "sep": None})
    return det, steps


def lu_steps(A):
    """Doolittle elimination with row swaps when a pivot is zero: P A = L U."""
    U = to_fractions(A)
    n = len(U)
    L = [[Fraction(0)] * n for _ in range(n)]
    perm = list(range(n))
    steps = []
    step(steps, "Start: U = A, L empty", U)
    for c in range(n):
        p = next((i for i in range(c, n) if U[i][c] != 0), None)
        if p is None:
            continue
        if p != c:
            U[c], U[p] = U[p], U[c]
            L[c], L[p] = L[p], L[c]
            perm[c], perm[p] = perm[p], perm[c]
            step(steps, f"Swap {row_name(c)} and {row_name(p)} (recorded in P)", U)
        for i in range(c + 1, n):
            f = U[i][c] / U[c][c]
            L[i][c] = f
            if f != 0:
                U[i] = [a - f * b for a, b in zip(U[i], U[c])]
                step(steps, f"{row_name(i)} -> {row_name(i)} - ({fmt(f)}) * {row_name(c)}; store l{i + 1}{c + 1} = {fmt(f)} in L", U)
    for i in range(n):
        L[i][i] = Fraction(1)
    P = [[Fraction(int(perm[i] == j)) for j in range(n)] for i in range(n)]
    steps.append({"text": "L holds the multipliers with 1s on the diagonal; U is the reduced matrix; P A = L U.",
                  "matrix": None, "sep": None})
    return P, L, U, steps


# ---------------------------------------------------------------- QR and SVD (floats)
def qr_steps(A):
    """Classical Gram-Schmidt on the columns. Needs independent columns (m >= n)."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q, R = np.zeros((m, n)), np.zeros((n, n))
    steps = []
    for j in range(n):
        v = A[:, j].copy()
        parts = []
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v -= R[i, j] * Q[:, i]
            parts.append(f"r{i + 1}{j + 1} = q{i + 1}.a{j + 1} = {fmt(R[i, j])}")
        R[j, j] = np.linalg.norm(v)
        if R[j, j] < 1e-10:
            steps.append({"text": f"Column {j + 1} depends on the earlier columns: Gram-Schmidt stops here.",
                          "matrix": None, "sep": None})
            return None, None, steps
        Q[:, j] = v / R[j, j]
        text = f"Column {j + 1}: " + ("; ".join(parts) + "; " if parts else "") + \
               f"subtract the projections, r{j + 1}{j + 1} = |v| = {fmt(R[j, j])}, q{j + 1} = v / r{j + 1}{j + 1}"
        steps.append({"text": text, "matrix": [[x] for x in Q[:, j]], "sep": None})
    return Q, R, steps


def svd_steps(A):
    """Singular values from the eigenvalues of A^T A, V from its eigenvectors, u_i = A v_i / s_i."""
    A = np.asarray(A, dtype=float)
    steps = []
    AtA = A.T @ A
    steps.append({"text": "Form A^T A (symmetric, so its eigenvalues are real and not negative)", "matrix": AtA.tolist(), "sep": None})
    lam, V = np.linalg.eigh(AtA)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    s = np.sqrt(np.clip(lam, 0, None))
    steps.append({"text": "Eigenvalues of A^T A, largest first: " + ", ".join(fmt(x) for x in lam) +
                          ". Singular values are their square roots: " + ", ".join(fmt(x) for x in s),
                  "matrix": None, "sep": None})
    steps.append({"text": "The eigenvectors of A^T A are the columns of V", "matrix": V.tolist(), "sep": None})
    U_cols = [A @ V[:, i] / s[i] for i in range(len(s)) if s[i] > 1e-10]
    if U_cols:
        steps.append({"text": "For each nonzero singular value, u_i = A v_i / s_i gives a column of U",
                      "matrix": np.array(U_cols).T.tolist(), "sep": None})
    return s, V, steps

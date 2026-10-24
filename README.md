# Ze Matrix: Advanced Linear Algebra Tool

**Live Demo:** [ze-matrix.streamlit.app](https://ze-matrix.streamlit.app)

![Preview](https://github.com/user-attachments/assets/a3730b49-dd6e-4434-b5d4-de4e81449abb)

An interactive linear algebra tool built with Streamlit. Supports advanced matrix operations, decompositions, equation solving, and an educational mode with quizzes.

## Features

### Matrix Operations
Addition · Subtraction · Multiplication · Transpose · Determinant · Inverse · Scalar Multiply · Power · Rank · Row Reduction (RREF) · Trace · Diagonal · Eigenvalues & Eigenvectors · Norm

### Decompositions
SVD · Cholesky · QR · LU

### Step-by-Step Solutions
Determinant, inverse, rank, row reduction and `Ax = b` are worked with exact fractions, one row operation at a time. LU shows each multiplier, QR each Gram-Schmidt step, and SVD how the singular values come from `A^T A`.

### Linear Equation Solver
Solve `Ax = b` by row reduction, with a clear answer for systems that have no solution or infinitely many.

### Educational Mode
- Concept guides with visual aids
- Interactive quizzes on 18 topics with a new random question each time and instant feedback
- Score per topic, saved to a file you can download and load again later

### Interface
- Matrices up to 8×8, with integers, decimals or fractions such as `1/3`
- Dark mode support
- LaTeX rendering for mathematical expressions

## Setup

```bash
git clone https://github.com/shemaiscard/ze-matrix.git
cd ze-matrix
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack
Python · Streamlit · NumPy · SciPy · Pandas

## License
MIT License

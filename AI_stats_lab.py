"""
AI_stats_lab.py

Autograded lab: Gradient Descent + Linear Regression (Diabetes)

You must implement the TODO functions below.
Do not change function names or return signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# =========================
# Helpers (you may use these)
# =========================

def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Add a bias (intercept) column of ones to X."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return np.hstack([np.ones((X.shape[0], 1)), X])


def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize using train statistics only.
    Returns: X_train_std, X_test_std, mean, std
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


@dataclass
class GDResult:
    theta: np.ndarray              # (d, )
    losses: np.ndarray             # (T, )
    thetas: np.ndarray             # (T, d) trajectory


# =========================
# Q1: Gradient descent + visualization data
# =========================

def gradient_descent_linreg(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 200,
    theta0: Optional[np.ndarray] = None,
) -> GDResult:
    """
    Linear regression with batch gradient descent on MSE loss.

    X should already include bias column if you want an intercept.

    Returns GDResult with final theta, per-epoch losses, and theta trajectory.

    The MSE loss is:
        L(theta) = (1/n) * sum((X @ theta - y)^2)

    The gradient with respect to theta is:
        grad = (2/n) * X^T (X @ theta - y)

    The update rule is:
        theta <- theta - lr * grad
    """
    n, d = X.shape
    y = y.reshape(-1)

    # Initialise theta (zeros unless provided)
    if theta0 is not None:
        theta = theta0.copy().astype(float)
    else:
        theta = np.zeros(d, dtype=float)

    losses = np.zeros(epochs)
    thetas = np.zeros((epochs, d))

    for t in range(epochs):
        # Forward pass: predictions
        y_pred = X @ theta                        # (n,)

        # Compute MSE loss
        residuals = y_pred - y                    # (n,)
        loss = float(np.mean(residuals ** 2))
        losses[t] = loss

        # Gradient of MSE w.r.t. theta: (2/n) X^T residuals
        grad = (2.0 / n) * (X.T @ residuals)     # (d,)

        # Parameter update
        theta = theta - lr * grad

        # Store trajectory (post-update theta for this epoch)
        thetas[t] = theta

    return GDResult(theta=theta, losses=losses, thetas=thetas)


def visualize_gradient_descent(
    lr: float = 0.1,
    epochs: int = 60,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Create a small synthetic 2D-parameter problem (bias + 1 feature),
    run gradient descent, and return data needed for visualization.

    Return dict with:
      - "theta_path": (T, 2) array of (theta0, theta1) over time
      - "losses": (T,) loss values
      - "X": design matrix used (with bias) shape (n, 2)
      - "y": targets shape (n,)

    Students can plot:
      - loss curve losses vs epoch
      - theta trajectory in parameter space (theta0 vs theta1)
    """
    rng = np.random.default_rng(seed)

    # Generate synthetic data: y = 1.5 + 2.5*x + noise
    n = 80
    x = rng.uniform(-1, 1, size=(n, 1))
    noise = rng.normal(0, 0.2, size=n)
    y = 1.5 + 2.5 * x.reshape(-1) + noise

    # Build design matrix with bias column
    X = add_bias_column(x)   # shape (n, 2)

    # Run gradient descent
    result = gradient_descent_linreg(X, y, lr=lr, epochs=epochs)

    return {
        "theta_path": result.thetas,   # (T, 2)
        "losses": result.losses,       # (T,)
        "X": X,                        # (n, 2)
        "y": y,                        # (n,)
    }


# =========================
# Q2: Diabetes regression using gradient descent
# =========================

def diabetes_linear_gd(
    lr: float = 0.05,
    epochs: int = 2000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Load diabetes dataset, split train/test, standardize, fit linear regression via GD.

    Steps:
      1. Load sklearn diabetes dataset
      2. Train/test split (stratified by index for reproducibility)
      3. Standardize features using train statistics only
      4. Add bias column
      5. Run gradient_descent_linreg
      6. Compute and return metrics

    Returns:
      train_mse, test_mse, train_r2, test_r2, theta
    """
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    # 1. Load data
    data = load_diabetes()
    X_raw, y_raw = data.data, data.target   # (442, 10), (442,)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=seed
    )

    # 3. Standardize features (using train statistics only)
    X_train_std, X_test_std, _, _ = standardize_train_test(X_train, X_test)

    # 4. Add bias column
    X_train_b = add_bias_column(X_train_std)
    X_test_b  = add_bias_column(X_test_std)

    # 5. Fit via gradient descent
    result = gradient_descent_linreg(X_train_b, y_train, lr=lr, epochs=epochs)
    theta = result.theta

    # 6. Compute metrics
    train_pred = X_train_b @ theta
    test_pred  = X_test_b  @ theta

    train_mse = mse(y_train, train_pred)
    test_mse  = mse(y_test,  test_pred)
    train_r2  = r2_score(y_train, train_pred)
    test_r2   = r2_score(y_test,  test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q3: Diabetes regression using analytical solution
# =========================

def diabetes_linear_analytical(
    ridge_lambda: float = 1e-8,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Closed-form solution (normal equation) for linear regression.

    Uses a tiny ridge term (lambda) for numerical stability:
      theta = (X^T X + lambda I)^{-1} X^T y

    Steps:
      1. Load diabetes, split, standardize (same pipeline as Q2)
      2. Add bias column
      3. Solve normal equations
      4. Return metrics

    Returns:
      train_mse, test_mse, train_r2, test_r2, theta
    """
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    # 1. Load and split
    data = load_diabetes()
    X_raw, y_raw = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=seed
    )

    # 2. Standardize
    X_train_std, X_test_std, _, _ = standardize_train_test(X_train, X_test)

    # 3. Add bias column
    X_train_b = add_bias_column(X_train_std)
    X_test_b  = add_bias_column(X_test_std)

    # 4. Normal equation with ridge regularisation for stability
    #    theta = (X^T X + lambda * I)^{-1} X^T y
    d = X_train_b.shape[1]
    A = X_train_b.T @ X_train_b + ridge_lambda * np.eye(d)
    b = X_train_b.T @ y_train
    theta = np.linalg.solve(A, b)    # more numerically stable than inv()

    # 5. Compute metrics
    train_pred = X_train_b @ theta
    test_pred  = X_test_b  @ theta

    train_mse = mse(y_train, train_pred)
    test_mse  = mse(y_test,  test_pred)
    train_r2  = r2_score(y_train, train_pred)
    test_r2   = r2_score(y_test,  test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q4: Compare GD vs analytical
# =========================

def diabetes_compare_gd_vs_analytical(
    lr: float = 0.05,
    epochs: int = 4000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Fit diabetes regression using both GD and analytical solution and compare.

    Return dict with:
      - "theta_l2_diff"      : L2 norm of (theta_gd - theta_analytical)
      - "train_mse_diff"     : |train_mse_gd - train_mse_analytical|
      - "test_mse_diff"      : |test_mse_gd  - test_mse_analytical|
      - "train_r2_diff"      : |train_r2_gd  - train_r2_analytical|
      - "test_r2_diff"       : |test_r2_gd   - test_r2_analytical|
      - "theta_cosine_sim"   : cosine similarity between the two theta vectors

    (Cosine similarity near 1 means parameters align.)
    """
    # Fit with gradient descent
    tr_mse_gd, te_mse_gd, tr_r2_gd, te_r2_gd, theta_gd = diabetes_linear_gd(
        lr=lr, epochs=epochs, test_size=test_size, seed=seed
    )

    # Fit with analytical solution
    tr_mse_an, te_mse_an, tr_r2_an, te_r2_an, theta_an = diabetes_linear_analytical(
        test_size=test_size, seed=seed
    )

    # L2 difference between parameter vectors
    theta_l2_diff = float(np.linalg.norm(theta_gd - theta_an))

    # Absolute metric differences
    train_mse_diff = float(abs(tr_mse_gd - tr_mse_an))
    test_mse_diff  = float(abs(te_mse_gd - te_mse_an))
    train_r2_diff  = float(abs(tr_r2_gd  - tr_r2_an))
    test_r2_diff   = float(abs(te_r2_gd  - te_r2_an))

    # Cosine similarity: dot(a, b) / (||a|| * ||b||)
    norm_gd = np.linalg.norm(theta_gd)
    norm_an = np.linalg.norm(theta_an)
    if norm_gd == 0 or norm_an == 0:
        theta_cosine_sim = 0.0
    else:
        theta_cosine_sim = float(
            np.dot(theta_gd, theta_an) / (norm_gd * norm_an)
        )

    return {
        "theta_l2_diff":    theta_l2_diff,
        "train_mse_diff":   train_mse_diff,
        "test_mse_diff":    test_mse_diff,
        "train_r2_diff":    train_r2_diff,
        "test_r2_diff":     test_r2_diff,
        "theta_cosine_sim": theta_cosine_sim,
    }

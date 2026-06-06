"""Rotosolve optimizer (Ostaszewski et al., 2021) for single-qubit-rotation
ansatze. For each parameter, energy is a sinusoid E(theta) = A*cos(theta+phi)+B;
sampling at {theta, theta+pi/2, theta-pi/2} gives the analytic minimizer."""
import math


def rotosolve(energy_fn, theta0, max_sweeps=50, tol=1e-8):
    """energy_fn: list[float] -> float. Returns (theta, energy, n_evals)."""
    theta = list(theta0)
    n_evals = 0
    last = energy_fn(theta)
    n_evals += 1
    for _ in range(max_sweeps):
        for j in range(len(theta)):
            base = theta[j]
            theta[j] = base
            e0 = energy_fn(theta)
            theta[j] = base + math.pi / 2
            ep = energy_fn(theta)
            theta[j] = base - math.pi / 2
            em = energy_fn(theta)
            n_evals += 3
            # Minimizer of A*cos(theta+phi)+B sampled at 0,+pi/2,-pi/2.
            theta[j] = base - math.pi / 2 - math.atan2(2 * e0 - ep - em, ep - em)
        e = energy_fn(theta)
        n_evals += 1
        if abs(e - last) < tol:
            return theta, e, n_evals
        last = e
    return theta, last, n_evals

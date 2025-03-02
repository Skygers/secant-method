import numpy as np

def analytical_solution(P1, P2, rho, g, h1, h2, v1):
    """
    Analytical solution for v2 from Bernoulli's equation:
    P1 + 0.5*rho*v1^2 + rho*g*h1 = P2 + 0.5*rho*v2^2 + rho*g*h2
    Solving for v2:
    v2 = sqrt((2/rho)*(P1 - P2 + rho*g*(h1 - h2) + 0.5*rho*v1^2))
    """
    try:
        term_under_sqrt = (2/rho) * (P1 - P2 + rho*g*(h1 - h2) + 0.5*rho*v1**2)
        if term_under_sqrt < 0:
            raise ValueError(f"No real solution exists (term under sqrt is negative: {term_under_sqrt:.2f})")
        return np.sqrt(term_under_sqrt)
    except Exception as e:
        raise ValueError(f"Error in analytical solution: {str(e)}")

def bernoulli_equation(v2, params):
    """
    Bernoulli equation function: P1 + 0.5*rho*v1^2 + rho*g*h1 = P2 + 0.5*rho*v2^2 + rho*g*h2
    Rearranged to be zero at the solution
    """
    P1, P2, rho, g, h1, h2, v1 = params

    # Scale the equation to improve numerical stability
    scale = abs(P1) if abs(P1) > 1 else 1.0

    # Scaled version of Bernoulli's equation
    return ((P1 + 0.5*rho*v1**2 + rho*g*h1) - 
            (P2 + 0.5*rho*v2**2 + rho*g*h2)) / scale

def secant_method(f, x0, x1, params, tol=1e-6, max_iter=100):
    """
    Implements the secant method for finding roots with improved stability
    """
    iterations = []

    # Validate initial guesses
    if x0 <= 0 or x1 <= 0:
        raise ValueError("Initial guesses must be positive")

    # If initial guesses are too close, adjust them
    if abs(x1 - x0) < tol:
        x1 = x0 * 1.1

    for i in range(max_iter):
        f_x0 = f(x0, params)
        f_x1 = f(x1, params)

        iterations.append((i+1, x1, f_x1))

        if abs(f_x1) < tol:
            return x1, iterations, True

        try:
            slope = (f_x1 - f_x0)/(x1 - x0)
            if abs(slope) < 1e-10:  # Avoid division by very small numbers
                return None, iterations, False

            x_new = x1 - f_x1 * (x1 - x0)/(f_x1 - f_x0)

            # Ensure the new guess is positive
            if x_new <= 0:
                x_new = (x0 + x1) / 2  # Use bisection step instead

        except ZeroDivisionError:
            return None, iterations, False

        x0, x1 = x1, x_new

        # Check if the method is oscillating
        if len(iterations) > 3:
            last_three = [abs(it[2]) for it in iterations[-3:]]
            if all(e > last_three[0] for e in last_three[1:]):
                return None, iterations, False

    return x1, iterations, False

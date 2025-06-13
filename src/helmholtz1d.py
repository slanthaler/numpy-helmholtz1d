import numpy as np
from scipy.linalg import solve_banded


def solve(wavespeed, frequency, L=1.0):
    '''
    Solve the 1D Helmholtz equation with variable wave speed and frequency.
    The Helmholtz equation is given by:
    .. math::
        -\frac{d^2 u}{dx^2} + k(x)^2 u = 0

    where :math:`k(x) = \frac{\omega}{c(x)}` is the wavenumber, with :math:`\omega = 2 \pi f` being the angular frequency.
    The boundary conditions are:
    - Dirichlet condition at the left boundary (source): :math:`u(0) = 1`
    - Outgoing (Sommerfeld) condition at the right boundary: :math:`\frac{du}{dx} + ik u = 0`

    Discretization by finite differences, leading to a tridiagonal system of equations.

    Args:
        wavespeed: A 1D array of wave speeds c(x) at each spatial point (equidistant grid).
        frequency: The frequency of the wave.
        L: Length of the domain (default is 1.0).
    Returns:
        u: The solution to the Helmholtz equation at each spatial point.
    '''
    # Discretize Domain and k(x)
    N = len(wavespeed) 
    dx = L / (N - 1) 
    omega = 2 * np.pi * frequency  # Angular frequency
    k = omega / wavespeed # Wavenumber at each spatial point

    # Construct the tridiagonal Matrix 'ab' and rhs vector 'b'
    # The structure for solve_banded is:
    # ab[0, :] = upper diagonal (shifted left, first element unused)
    # ab[1, :] = main diagonal
    # ab[2, :] = lower diagonal (shifted right, last element unused)
    ab = np.zeros((3, N), dtype=np.complex128) # 3 rows for super, main, sub diagonals
    b = np.zeros(N, dtype=np.complex128)

    # Equation: u[i-1] + (-2 + (k[i] dx)^2) u[i] + u[i+1] = 0
    main_diag_coeffs = -2 + (k*dx)**2
    ab[0, :] = 1.0
    ab[1, :] = main_diag_coeffs
    ab[2, :] = 1.0

    # Apply Boundary Conditions
    # Left Boundary (i=0) - Dirichlet: u[0] = u_left
    u_left = 1.0  # Dirichlet condition at the left boundary
    ab[1, 0] = 1.0 # Main diagonal element for u[0]
    ab[0, 1] = 0.0 # Unused super-diagonal element
    b[0] = u_left

    # Right Boundary - Outgoing (Sommerfeld)
    # Equation: u[N-2] + (i k[N-1] dx - 1) u[N-1] = 0
    ab[1,-1] = 1j*k[-1]*dx - 1 # Main diagonal element for u[N-1]
    ab[2,-1] = 1.0 # Coefficient of u[N-2] in the last equation
    b[-1] = 0.0 # No source term for outgoing boundary

    # Solve the tridiagonal system using solve_banded
    u = solve_banded((1, 1), ab, b)

    return u

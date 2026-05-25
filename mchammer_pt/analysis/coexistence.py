"""First-order coexistence-point analysis on stitched Wang-Landau DOS.

Locates the equal-area coexistence temperature T_c at which the
canonical energy distribution P(E|T) has equal integrated weight on
either side of the free-energy minimum E*(T) between its two peaks.
"""
from __future__ import annotations


class NotBimodalError(ValueError):
    """Raised when the DOS does not show two well-separated peaks.

    Carries a message naming the temperature at which detection ran
    and what shape was found (one peak / adjacent peaks / no interior
    minimum).
    """


class NoBracketError(ValueError):
    """Raised when bisection cannot proceed.

    Either ``imbalance(T)`` does not change sign across the supplied
    or auto-built ``T_bracket``, or shape analysis failed at one of
    the bracket endpoints (or mid-bracket — the bracket extends
    outside the bimodal region).
    """


def _parabolic_vertex(
    x_l: float, x_c: float, x_r: float,
    y_l: float, y_c: float, y_r: float,
) -> float:
    """Return the x-coordinate of the parabolic vertex through three samples.

    Fits ``y = a x^2 + b x + c`` to the three points by Lagrange
    interpolation and returns ``-b / (2 a)``. In typical use the three
    points are the centre bin of an extremum of ``phi(E)`` and its two
    neighbours on the DOS energy grid; the function returns a sub-bin
    refined position of the extremum.

    Falls back to ``x_c`` in two cases:

    - ``denom == 0``: two of the three x-values coincide. Cannot arise
      from distinct bin centres on a uniform grid; this branch is a
      defensive guard.
    - ``a == 0``: the three points are collinear (the local fit is
      linear, no parabolic vertex exists). Returning the centre sample
      is the natural no-op.
    """
    denom = (x_l - x_c) * (x_l - x_r) * (x_c - x_r)
    if denom == 0.0:
        return x_c
    a = (
        x_r * (y_c - y_l)
        + x_c * (y_l - y_r)
        + x_l * (y_r - y_c)
    ) / denom
    b = (
        x_r * x_r * (y_l - y_c)
        + x_c * x_c * (y_r - y_l)
        + x_l * x_l * (y_c - y_r)
    ) / denom
    if a == 0.0:
        return x_c
    return -b / (2.0 * a)

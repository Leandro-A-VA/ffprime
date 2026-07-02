"""
Electrostatic potential and electric field from atom-centered multipoles.

Implements monopole, dipole, and quadrupole contributions following the
multipole expansion of the electrostatic potential in atomic units.

References
----------
Stone, A.J. "The Theory of Intermolecular Forces", Oxford University Press.
"""

import numpy as np


def _validate_shapes(coords, points):
    """Validate coordinate and point array shapes."""
    coords = np.asarray(coords)
    points = np.asarray(points)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coords must have shape (N, 3), got {coords.shape}"
        )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape (M, 3), got {points.shape}"
        )


def monopole_potential(charges, coords, points):
    """
    Compute electrostatic potential from atomic monopoles (charges).
    V(r) = sum_i q_i / |r - r_i|

    Parameters
    ----------
    charges : np.ndarray, shape (N,)
        Atomic charges in atomic units.
    coords : np.ndarray, shape (N, 3)
        Atomic coordinates in atomic units.
    points : np.ndarray, shape (M, 3)
        Field points at which to evaluate the potential.

    Returns
    -------
    potential : np.ndarray, shape (M,)
        Electrostatic potential at each field point in atomic units.
    """
    charges = np.asarray(charges)
    coords = np.asarray(coords)
    points = np.asarray(points)
    _validate_shapes(coords, points)

    if len(charges) != len(coords):
        raise ValueError("charges and coords must have same length")

    r_vecs = points[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r = np.linalg.norm(r_vecs, axis=-1)
    safe_r = np.where(r < 1e-12, np.inf, r)
    return np.sum(charges[np.newaxis, :] / safe_r, axis=1)


def monopole_field(charges, coords, points):
    """
    Compute electric field from atomic monopoles (charges).
    E(r) = sum_i q_i (r - r_i) / |r - r_i|^3

    Parameters
    ----------
    charges : np.ndarray, shape (N,)
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    field : np.ndarray, shape (M, 3)
    """
    charges = np.asarray(charges)
    coords = np.asarray(coords)
    points = np.asarray(points)
    _validate_shapes(coords, points)

    if len(charges) != len(coords):
        raise ValueError("charges and coords must have same length")

    r_vecs = points[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r = np.linalg.norm(r_vecs, axis=-1)
    safe_r = np.where(r < 1e-12, np.inf, r)
    field = charges[np.newaxis, :, np.newaxis] * r_vecs / safe_r[:, :, np.newaxis] ** 3
    return np.sum(field, axis=1)


def dipole_potential(dipoles, coords, points):
    """
    Compute electrostatic potential from atomic dipoles.

    V(r) = sum_i [ p_i . (r - r_i) ] / |r - r_i|^3

    Parameters
    ----------
    dipoles : np.ndarray, shape (N, 3)
        Atomic dipole moment vectors in atomic units.
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    potential : np.ndarray, shape (M,)
    """
    dipoles = np.asarray(dipoles)
    coords = np.asarray(coords)
    points = np.asarray(points)
    _validate_shapes(coords, points)

    if len(dipoles) != len(coords):
        raise ValueError("dipoles and coords must have same length")

    if dipoles.ndim != 2 or dipoles.shape[1] != 3:
        raise ValueError(
            f"dipoles must have shape (N, 3), got {dipoles.shape}"
        )

    r_vecs = points[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r = np.linalg.norm(r_vecs, axis=-1)
    safe_r = np.where(r < 1e-12, np.inf, r)
    p_dot_r = np.einsum("ij,mij->mi", dipoles, r_vecs)
    return np.sum(p_dot_r / safe_r ** 3, axis=1)


def dipole_field(dipoles, coords, points):
    """
    Compute electric field from atomic dipoles.

    E(r) = sum_i [ 3(p_i . r̂)r̂ - p_i ] / |r - r_i|^3

    Parameters
    ----------
    dipoles : np.ndarray, shape (N, 3)
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    field : np.ndarray, shape (M, 3)
    """
    dipoles = np.asarray(dipoles)
    coords = np.asarray(coords)
    points = np.asarray(points)
    _validate_shapes(coords, points)

    if len(dipoles) != len(coords):
        raise ValueError("dipoles and coords must have same length")

    if dipoles.ndim != 2 or dipoles.shape[1] != 3:
        raise ValueError(
            f"dipoles must have shape (N, 3), got {dipoles.shape}"
        )

    r_vecs = points[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r = np.linalg.norm(r_vecs, axis=-1)
    safe_r = np.where(r < 1e-12, np.inf, r)
    r_hat = r_vecs / safe_r[:, :, np.newaxis]
    p_dot_rhat = np.einsum("ij,mij->mi", dipoles, r_hat)
    term = (3 * p_dot_rhat[:, :, np.newaxis] * r_hat
            - dipoles[np.newaxis, :, :])
    return np.sum(term / safe_r[:, :, np.newaxis] ** 3, axis=1)


def quadrupole_potential(quadrupoles, coords, points):
    """
    Compute electrostatic potential from atomic quadrupoles (traceless).

    V(r) = sum_i [ Q_i:rr ] / (|r - r_i|^5)

    Parameters
    ----------
    quadrupoles : np.ndarray, shape (N, 3, 3)
        Traceless quadrupole moment tensors in atomic units.
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    potential : np.ndarray, shape (M,)
    """
    quadrupoles = np.asarray(quadrupoles)
    coords = np.asarray(coords)
    points = np.asarray(points)
    _validate_shapes(coords, points)

    if len(quadrupoles) != len(coords):
        raise ValueError("quadrupoles and coords must have same length")

    if quadrupoles.ndim != 3 or quadrupoles.shape[1:] != (3, 3):
        raise ValueError(
            f"quadrupoles must have shape (N, 3, 3), got {quadrupoles.shape}"
        )

    r_vecs = points[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r = np.linalg.norm(r_vecs, axis=-1)
    safe_r = np.where(r < 1e-12, np.inf, r)
    Qrr = np.einsum("nab,mna,mnb->mn", quadrupoles, r_vecs, r_vecs)
    return np.sum(Qrr / (safe_r ** 5), axis=1)


def quadrupole_field(quadrupoles, coords, points):
    """
    Compute electric field from atomic quadrupoles (traceless).

    E(r) = sum_i [ 5*(Q_i:rr)*r / (r^7) - (Q_i.r) / r^5 ]

    Parameters
    ----------
    quadrupoles : np.ndarray, shape (N, 3, 3)
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    field : np.ndarray, shape (M, 3)
    """
    quadrupoles = np.asarray(quadrupoles)
    coords = np.asarray(coords)
    points = np.asarray(points)
    _validate_shapes(coords, points)

    if len(quadrupoles) != len(coords):
        raise ValueError("quadrupoles and coords must have same length")

    if quadrupoles.ndim != 3 or quadrupoles.shape[1:] != (3, 3):
        raise ValueError(
            f"quadrupoles must have shape (N, 3, 3), got {quadrupoles.shape}"
        )

    r_vecs = points[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r = np.linalg.norm(r_vecs, axis=-1)
    safe_r = np.where(r < 1e-12, np.inf, r)
    Qrr = np.einsum("nab,mna,mnb->mn", quadrupoles, r_vecs, r_vecs)
    Qr = np.einsum("nab,mnb->mna", quadrupoles, r_vecs)
    term1 = (5 * Qrr[:, :, np.newaxis] * r_vecs
             / (safe_r[:, :, np.newaxis] ** 7))
    term2 = Qr / safe_r[:, :, np.newaxis] ** 5
    return np.sum(term1 - term2, axis=1)


def total_potential(coords, points, charges=None, dipoles=None, quadrupoles=None):
    """
    Compute total electrostatic potential from all multipole contributions.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)
    charges : np.ndarray, shape (N,), optional
    dipoles : np.ndarray, shape (N, 3), optional
    quadrupoles : np.ndarray, shape (N, 3, 3), optional

    Returns
    -------
    potential : np.ndarray, shape (M,)
    """
    coords = np.asarray(coords)
    points = np.asarray(points)
    _validate_shapes(coords, points)

    if charges is not None and len(charges) != len(coords):
        raise ValueError("charges and coords must have same length")

    if dipoles is not None and len(dipoles) != len(coords):
        raise ValueError("dipoles and coords must have same length")

    if quadrupoles is not None and len(quadrupoles) != len(coords):
        raise ValueError("quadrupoles and coords must have same length")

    potential = np.zeros(points.shape[0])
    if charges is not None:
        potential += monopole_potential(charges, coords, points)
    if dipoles is not None:
        potential += dipole_potential(dipoles, coords, points)
    if quadrupoles is not None:
        potential += quadrupole_potential(quadrupoles, coords, points)
    return potential


def total_field(coords, points, charges=None, dipoles=None, quadrupoles=None):
    """
    Compute total electric field from all multipole contributions.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)
    charges : np.ndarray, shape (N,), optional
    dipoles : np.ndarray, shape (N, 3), optional
    quadrupoles : np.ndarray, shape (N, 3, 3), optional

    Returns
    -------
    field : np.ndarray, shape (M, 3)
    """
    coords = np.asarray(coords)
    points = np.asarray(points)
    _validate_shapes(coords, points)

    if charges is not None and len(charges) != len(coords):
        raise ValueError("charges and coords must have same length")

    if dipoles is not None and len(dipoles) != len(coords):
        raise ValueError("dipoles and coords must have same length")

    if quadrupoles is not None and len(quadrupoles) != len(coords):
        raise ValueError("quadrupoles and coords must have same length")

    field = np.zeros((points.shape[0], 3))
    if charges is not None:
        field += monopole_field(charges, coords, points)
    if dipoles is not None:
        field += dipole_field(dipoles, coords, points)
    if quadrupoles is not None:
        field += quadrupole_field(quadrupoles, coords, points)
    return field
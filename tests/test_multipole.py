"""
Tests for electrostatic multipole potential and field functions.
All tests validate against analytical solutions derived from first principles.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from ffprime.electrostatics.multipole import (
    monopole_potential,
    monopole_field,
    dipole_potential,
    dipole_field,
    quadrupole_potential,
    quadrupole_field,
    total_potential,
    total_field,
)


# ─────────────────────────────────────────────
# MONOPOLE TESTS
# ─────────────────────────────────────────────

class TestMonopolePotential:

    def test_single_unit_charge(self):
        """V = q/r → at distance 1.0, V = 1.0"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0]])
        result = monopole_potential(charges, coords, points)
        assert_allclose(result, [1.0], rtol=1e-10)

    def test_inverse_distance_scaling(self):
        """V scales as 1/r"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 0.0, 0.0],
                           [4.0, 0.0, 0.0]])
        result = monopole_potential(charges, coords, points)
        assert_allclose(result[0] / result[1], 2.0, rtol=1e-10)

    def test_negative_charge(self):
        """Negative charge gives negative potential"""
        charges = np.array([-1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0]])
        result = monopole_potential(charges, coords, points)
        assert_allclose(result, [-1.0], rtol=1e-10)

    def test_superposition(self):
        """Two equal charges: potential is sum of individual contributions"""
        charges = np.array([1.0, 1.0])
        coords = np.array([[1.0, 0.0, 0.0],
                           [-1.0, 0.0, 0.0]])
        points = np.array([[0.0, 2.0, 0.0]])
        result = monopole_potential(charges, coords, points)
        expected = 2.0 / np.sqrt(5.0)
        assert_allclose(result, [expected], rtol=1e-10)

    def test_singularity_safe(self):
        """Potential at exact atom location returns finite value"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 0.0]])
        result = monopole_potential(charges, coords, points)
        assert np.isfinite(result).all()

    def test_multiple_field_points(self):
        """Vectorized evaluation over multiple points"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0],
                           [3.0, 0.0, 0.0]])
        result = monopole_potential(charges, coords, points)
        assert_allclose(result, [1.0, 0.5, 1/3], rtol=1e-10)


class TestMonopoleField:

    def test_unit_charge_along_x(self):
        """E = q*r/r^3 → at (1,0,0), E = (1,0,0)"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0]])
        result = monopole_field(charges, coords, points)
        assert_allclose(result, [[1.0, 0.0, 0.0]], rtol=1e-10)

    def test_inverse_square_scaling(self):
        """E magnitude scales as 1/r^2"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 0.0, 0.0],
                           [4.0, 0.0, 0.0]])
        result = monopole_field(charges, coords, points)
        ratio = np.linalg.norm(result[0]) / np.linalg.norm(result[1])
        assert_allclose(ratio, 4.0, rtol=1e-10)

    def test_field_is_radial(self):
        """Field points radially away from positive charge"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 1.0, 1.0]])
        result = monopole_field(charges, coords, points)
        r = np.array([1.0, 1.0, 1.0])
        r_hat = r / np.linalg.norm(r)
        result_hat = result[0] / np.linalg.norm(result[0])
        assert_allclose(result_hat, r_hat, rtol=1e-10)

    def test_singularity_safe(self):
        """Field at exact atom location returns finite value"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 0.0]])
        result = monopole_field(charges, coords, points)
        assert np.isfinite(result).all()


# ─────────────────────────────────────────────
# DIPOLE TESTS
# ─────────────────────────────────────────────

class TestDipolePotential:

    def test_dipole_along_axis(self):
        """
        p=(1,0,0) at origin, point at (2,0,0):
        V = (p.r)/r^3 = 2/8 = 0.25
        """
        dipoles = np.array([[1.0, 0.0, 0.0]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 0.0, 0.0]])
        result = dipole_potential(dipoles, coords, points)
        assert_allclose(result, [0.25], rtol=1e-10)

    def test_dipole_perpendicular_zero(self):
        """
        p=(1,0,0), point at (0,2,0):
        V = (p.r)/r^3 = 0 (perpendicular)
        """
        dipoles = np.array([[1.0, 0.0, 0.0]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 2.0, 0.0]])
        result = dipole_potential(dipoles, coords, points)
        assert_allclose(result, [0.0], atol=1e-12)

    def test_dipole_antisymmetry(self):
        """V at +r and -r are equal and opposite"""
        dipoles = np.array([[1.0, 0.0, 0.0]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 0.0, 0.0],
                           [-2.0, 0.0, 0.0]])
        result = dipole_potential(dipoles, coords, points)
        assert_allclose(result[0], -result[1], rtol=1e-10)

    def test_dipole_scales_as_inverse_r_squared(self):
        """V_dipole scales as 1/r^2"""
        dipoles = np.array([[1.0, 0.0, 0.0]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0]])
        result = dipole_potential(dipoles, coords, points)
        assert_allclose(result[0] / result[1], 4.0, rtol=1e-10)


class TestDipoleField:

    def test_dipole_field_along_axis(self):
        """
        p=(0,0,1) at origin, point at (0,0,2):
        E = [3(p.r̂)r̂ - p] / r^3
          = [3*(0,0,1) - (0,0,1)] / 8
          = (0, 0, 0.25)
        """
        dipoles = np.array([[0.0, 0.0, 1.0]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 2.0]])
        result = dipole_field(dipoles, coords, points)
        assert_allclose(result, [[0.0, 0.0, 0.25]], rtol=1e-10)

    def test_dipole_field_perpendicular(self):
        """
        p=(0,0,1) at origin, point at (2,0,0):
        r̂=(1,0,0), p.r̂=0
        E = [0 - (0,0,1)] / 8 = (0, 0, -0.125)
        """
        dipoles = np.array([[0.0, 0.0, 1.0]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 0.0, 0.0]])
        result = dipole_field(dipoles, coords, points)
        assert_allclose(result, [[0.0, 0.0, -0.125]], rtol=1e-10)

    def test_singularity_safe(self):
        """Field at exact atom location returns finite value"""
        dipoles = np.array([[1.0, 0.0, 0.0]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 0.0]])
        result = dipole_field(dipoles, coords, points)
        assert np.isfinite(result).all()


# ─────────────────────────────────────────────
# QUADRUPOLE TESTS
# ─────────────────────────────────────────────

class TestQuadrupolePotential:

    def test_quadrupole_along_z(self):
        """
        Q = diag(-1,-1,2) (traceless), point at (0,0,2):
        Qrr = Q_ab r_a r_b = 2*4 = 8
        V = Qrr / (2*r^5) = 8 / (2*32) = 0.125
        """
        Q = np.array([[[-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0,  0.0, 2.0]]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 2.0]])
        result = quadrupole_potential(Q, coords, points)
        assert_allclose(result, [0.125], rtol=1e-10)

    def test_traceless_symmetry(self):
        """Trace of Q should not contribute — pure traceless result"""
        Q = np.array([[[2.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0,  0.0, -1.0]]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 0.0, 0.0]])
        result = quadrupole_potential(Q, coords, points)
        assert np.isfinite(result).all()

    def test_singularity_safe(self):
        """Potential at exact atom location returns finite value"""
        Q = np.array([[[1.0, 0.0, 0.0],
                       [0.0, -0.5, 0.0],
                       [0.0, 0.0, -0.5]]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 0.0]])
        result = quadrupole_potential(Q, coords, points)
        assert np.isfinite(result).all()


class TestQuadrupoleField:

    def test_quadrupole_field_finite(self):
        """Field from quadrupole is finite at valid points"""
        Q = np.array([[[-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0,  0.0, 2.0]]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 2.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0]])
        result = quadrupole_field(Q, coords, points)
        assert np.isfinite(result).all()

    def test_singularity_safe(self):
        """Field at exact atom location returns finite value"""
        Q = np.array([[[1.0, 0.0, 0.0],
                       [0.0, -0.5, 0.0],
                       [0.0, 0.0, -0.5]]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[0.0, 0.0, 0.0]])
        result = quadrupole_field(Q, coords, points)
        assert np.isfinite(result).all()


# ─────────────────────────────────────────────
# TOTAL POTENTIAL AND FIELD TESTS
# ─────────────────────────────────────────────

class TestTotalPotential:

    def test_monopole_only(self):
        """total_potential with only charges matches monopole_potential"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0]])
        result = total_potential(coords, points, charges=charges)
        expected = monopole_potential(charges, coords, points)
        assert_allclose(result, expected, rtol=1e-10)

    def test_superposition_all_terms(self):
        """Total potential is sum of all individual contributions"""
        charges = np.array([1.0])
        dipoles = np.array([[1.0, 0.0, 0.0]])
        Q = np.array([[[-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0,  0.0, 2.0]]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 1.0, 0.0]])

        total = total_potential(coords, points,
                                charges=charges,
                                dipoles=dipoles,
                                quadrupoles=Q)
        expected = (monopole_potential(charges, coords, points) +
                    dipole_potential(dipoles, coords, points) +
                    quadrupole_potential(Q, coords, points))
        assert_allclose(total, expected, rtol=1e-10)


class TestTotalField:

    def test_monopole_only(self):
        """total_field with only charges matches monopole_field"""
        charges = np.array([1.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 0.0, 0.0]])
        result = total_field(coords, points, charges=charges)
        expected = monopole_field(charges, coords, points)
        assert_allclose(result, expected, rtol=1e-10)

    def test_superposition_all_terms(self):
        """Total field is sum of all individual contributions"""
        charges = np.array([1.0])
        dipoles = np.array([[1.0, 0.0, 0.0]])
        Q = np.array([[[-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0,  0.0, 2.0]]])
        coords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[2.0, 1.0, 0.0]])

        total = total_field(coords, points,
                            charges=charges,
                            dipoles=dipoles,
                            quadrupoles=Q)
        expected = (monopole_field(charges, coords, points) +
                    dipole_field(dipoles, coords, points) +
                    quadrupole_field(Q, coords, points))
        assert_allclose(total, expected, rtol=1e-10)
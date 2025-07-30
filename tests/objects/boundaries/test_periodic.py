from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from jax import random

from fdtdx.objects.boundaries.periodic import PeriodicBoundary, PeriodicBoundaryState
from fdtdx.typing import SliceTuple3D

if TYPE_CHECKING:
    from .conftest import MockSimulationConfig


@pytest.fixture
def periodic_boundary(
    mock_simulation_config: "MockSimulationConfig", grid_slice_tuple: SliceTuple3D
) -> PeriodicBoundary:
    """Fixture to create a PeriodicBoundary instance."""
    pbc = PeriodicBoundary(axis=0, direction="+")
    pbc = pbc.aset("_grid_slice_tuple", grid_slice_tuple)
    return pbc.aset("_config", mock_simulation_config, create_new_ok=True)


def test_descriptive_name(periodic_boundary: PeriodicBoundary):
    """Test the descriptive_name property."""
    assert periodic_boundary.descriptive_name == "max_x"


def test_thickness(periodic_boundary: PeriodicBoundary):
    """Test the thickness property."""
    assert periodic_boundary.thickness == 1


def test_init_state(periodic_boundary: PeriodicBoundary):
    """Test the init_state method."""
    state = periodic_boundary.init_state()
    assert isinstance(state, PeriodicBoundaryState)
    assert state.E_opposite.shape == (3, 10, 20, 30)
    assert state.H_opposite.shape == (3, 10, 20, 30)


def test_reset_state(periodic_boundary: PeriodicBoundary):
    """Test the reset_state method."""
    state = periodic_boundary.init_state()
    reset_state = periodic_boundary.reset_state(state)
    assert jnp.all(reset_state.E_opposite == 0)
    assert jnp.all(reset_state.H_opposite == 0)


def test_boundary_slice(periodic_boundary: PeriodicBoundary):
    """Test the boundary_slice property."""
    boundary_slice = periodic_boundary.boundary_slice
    assert boundary_slice == (slice(0, 1), slice(0, 20), slice(0, 30))


def test_opposite_slice(periodic_boundary: PeriodicBoundary):
    """Test the opposite_slice property."""
    opposite_slice = periodic_boundary.opposite_slice
    assert opposite_slice == (slice(9, 10), slice(0, 20), slice(0, 30))


def test_update_E_boundary_state(periodic_boundary: PeriodicBoundary):
    """Test the update_E_boundary_state method."""
    state = periodic_boundary.init_state()
    H = jnp.ones((3, 10, 20, 30))
    updated_state = periodic_boundary.update_E_boundary_state(state, H)
    assert jnp.all(updated_state.H_opposite == H[..., 9:10, :, :])


def test_update_H_boundary_state(periodic_boundary: PeriodicBoundary):
    """Test the update_H_boundary_state method."""
    state = periodic_boundary.init_state()
    E = jnp.ones((3, 10, 20, 30))
    updated_state = periodic_boundary.update_H_boundary_state(state, E)
    assert jnp.all(updated_state.E_opposite == E[..., 9:10, :, :])


def test_update_E(periodic_boundary: PeriodicBoundary):
    """Test the update_E method."""
    state = periodic_boundary.init_state()
    key = random.PRNGKey(42)
    E = random.uniform(key, shape=(3, 10, 20, 30))
    updated_E = periodic_boundary.update_E(E, state, inverse_permittivity=jnp.ones((3, 10, 20, 30)))
    assert jnp.all(updated_E[..., 0:1, :, :] == E[..., 9:10, :, :])


def test_update_H(periodic_boundary: PeriodicBoundary):
    """Test the update_H method."""
    state = periodic_boundary.init_state()
    key = random.PRNGKey(43)
    H = random.uniform(key, shape=(3, 10, 20, 30))
    updated_H = periodic_boundary.update_H(H, state, inverse_permeability=1.0)
    assert jnp.all(updated_H[..., 0:1, :, :] == H[..., 9:10, :, :])

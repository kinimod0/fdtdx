from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from jax import random

from fdtdx.objects.boundaries.perfectly_matched_layer import (
    PerfectlyMatchedLayer,
    PMLBoundaryState,
)
from fdtdx.typing import SliceTuple3D

if TYPE_CHECKING:
    from .conftest import MockSimulationConfig


@pytest.fixture
def pml_boundary(
    mock_simulation_config: "MockSimulationConfig", grid_slice_tuple: SliceTuple3D
) -> PerfectlyMatchedLayer:
    """Fixture to create a PerfectlyMatchedLayer instance."""
    pml = PerfectlyMatchedLayer(axis=0, direction="+")
    pml = pml.aset("_grid_slice_tuple", grid_slice_tuple)
    return pml.aset("_config", mock_simulation_config, create_new_ok=True)


@pytest.fixture
def pml_boundary_small(
    mock_simulation_config: "MockSimulationConfig", grid_slice_tuple_small: SliceTuple3D
) -> PerfectlyMatchedLayer:
    """Fixture to create a PerfectlyMatchedLayer instance."""
    pml = PerfectlyMatchedLayer(axis=0, direction="+")
    pml = pml.aset("_grid_slice_tuple", grid_slice_tuple_small)
    return pml.aset("_config", mock_simulation_config, create_new_ok=True)


def test_descriptive_name(pml_boundary: PerfectlyMatchedLayer):
    """Test the descriptive_name property."""
    assert pml_boundary.descriptive_name == "max_x"


def test_thickness(pml_boundary: PerfectlyMatchedLayer):
    """Test the thickness property."""
    assert pml_boundary.thickness == 10


def test_init_state(pml_boundary: PerfectlyMatchedLayer):
    """Test the init_state method."""
    state = pml_boundary.init_state()
    assert isinstance(state, PMLBoundaryState)
    assert state.psi_Ex.shape == (3, 10, 20, 30)
    assert state.psi_Ey.shape == (3, 10, 20, 30)
    assert state.psi_Ez.shape == (3, 10, 20, 30)
    assert state.psi_Hx.shape == (3, 10, 20, 30)
    assert state.psi_Hy.shape == (3, 10, 20, 30)
    assert state.psi_Hz.shape == (3, 10, 20, 30)


def test_reset_state(pml_boundary: PerfectlyMatchedLayer):
    """Test the reset_state method."""
    state = pml_boundary.init_state()
    reset_state = pml_boundary.reset_state(state)
    assert jnp.all(reset_state.psi_Ex == 0)
    assert jnp.all(reset_state.psi_Ey == 0)
    assert jnp.all(reset_state.psi_Ez == 0)
    assert jnp.all(reset_state.psi_Hx == 0)
    assert jnp.all(reset_state.psi_Hy == 0)
    assert jnp.all(reset_state.psi_Hz == 0)


def test_update_E_boundary_state(pml_boundary: PerfectlyMatchedLayer):
    """Test the update_E_boundary_state method."""
    state = pml_boundary.init_state()
    H = jnp.ones((3, 10, 20, 30))
    updated_state = pml_boundary.update_E_boundary_state(state, H)
    assert updated_state.psi_Ex.shape == state.psi_Ex.shape
    assert updated_state.psi_Ey.shape == state.psi_Ey.shape
    assert updated_state.psi_Ez.shape == state.psi_Ez.shape


def test_update_H_boundary_state(pml_boundary: PerfectlyMatchedLayer):
    """Test the update_H_boundary_state method."""
    state = pml_boundary.init_state()
    E = jnp.ones((3, 10, 20, 30))
    updated_state = pml_boundary.update_H_boundary_state(state, E)
    assert updated_state.psi_Hx.shape == state.psi_Hx.shape
    assert updated_state.psi_Hy.shape == state.psi_Hy.shape
    assert updated_state.psi_Hz.shape == state.psi_Hz.shape


def test_update_E(pml_boundary: PerfectlyMatchedLayer):
    """Test the update_E method."""
    state = pml_boundary.init_state()
    key = random.PRNGKey(42)
    E = random.uniform(key, shape=(3, 10, 20, 30))
    inverse_permittivity = jnp.ones((10, 20, 30))
    updated_E = pml_boundary.update_E(E, state, inverse_permittivity)
    assert updated_E.shape == E.shape


def test_update_H(pml_boundary: PerfectlyMatchedLayer):
    """Test the update_H method."""
    state = pml_boundary.init_state()
    key = random.PRNGKey(43)
    H = random.uniform(key, shape=(3, 10, 20, 30))
    inverse_permeability = jnp.ones((10, 20, 30))
    updated_H = pml_boundary.update_H(H, state, inverse_permeability)
    assert updated_H.shape == H.shape


def test_update_H_small(pml_boundary_small: PerfectlyMatchedLayer):
    """Test the update_H method."""
    state = pml_boundary_small.init_state()
    H = jnp.ones((3, *pml_boundary_small.grid_shape))
    inverse_permeability = jnp.ones(pml_boundary_small.grid_shape)
    updated_H = pml_boundary_small.update_H(H, state, inverse_permeability)
    assert jnp.allclose(updated_H, jnp.array([1, 0.8, 2 / 3])[None, :, None, None])

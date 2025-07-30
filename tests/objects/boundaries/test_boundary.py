import jax.numpy as jnp
import pytest

from fdtdx.objects.boundaries.boundary import BaseBoundary, BaseBoundaryState
from fdtdx.typing import SliceTuple3D


class MockBoundaryState(BaseBoundaryState):
    """Mock implementation of BaseBoundaryState for testing."""

    pass


class MockBoundary(BaseBoundary[MockBoundaryState]):
    """Mock implementation of BaseBoundary for testing."""

    @property
    def descriptive_name(self) -> str:
        return "mock_boundary"

    @property
    def thickness(self) -> int:
        return 1

    def init_state(self) -> MockBoundaryState:
        return MockBoundaryState()

    def reset_state(self, state: MockBoundaryState) -> MockBoundaryState:
        return state

    def update_E_boundary_state(self, boundary_state: MockBoundaryState, H: jnp.ndarray) -> MockBoundaryState:
        return boundary_state

    def update_H_boundary_state(self, boundary_state: MockBoundaryState, E: jnp.ndarray) -> MockBoundaryState:
        return boundary_state

    def update_E(
        self, E: jnp.ndarray, boundary_state: MockBoundaryState, inverse_permittivity: jnp.ndarray
    ) -> jnp.ndarray:
        return E

    def update_H(
        self, H: jnp.ndarray, boundary_state: MockBoundaryState, inverse_permeability: jnp.ndarray | float
    ) -> jnp.ndarray:
        return H


@pytest.fixture
def mock_boundary(grid_slice_tuple: SliceTuple3D) -> MockBoundary:
    """Fixture to create a mock boundary instance."""
    _mock_boundary = MockBoundary(axis=0, direction="+")
    return _mock_boundary.aset("_grid_slice_tuple", grid_slice_tuple)


def test_descriptive_name(mock_boundary: MockBoundary):
    """Test the descriptive_name property."""
    assert mock_boundary.descriptive_name == "mock_boundary"


def test_thickness(mock_boundary: MockBoundary):
    """Test the thickness property."""
    assert mock_boundary.thickness == 1


def test_init_state(mock_boundary: MockBoundary):
    """Test the init_state method."""
    state = mock_boundary.init_state()
    assert isinstance(state, MockBoundaryState)


def test_reset_state(mock_boundary: MockBoundary):
    """Test the reset_state method."""
    state = mock_boundary.init_state()
    reset_state = mock_boundary.reset_state(state)
    assert reset_state == state


def test_update_E_boundary_state(mock_boundary: MockBoundary):
    """Test the update_E_boundary_state method."""
    state = mock_boundary.init_state()
    H = jnp.zeros((3, 3, 3))
    updated_state = mock_boundary.update_E_boundary_state(state, H)
    assert updated_state == state


def test_update_H_boundary_state(mock_boundary: MockBoundary):
    """Test the update_H_boundary_state method."""
    state = mock_boundary.init_state()
    E = jnp.zeros((3, 3, 3))
    updated_state = mock_boundary.update_H_boundary_state(state, E)
    assert updated_state == state


def test_update_E(mock_boundary: MockBoundary):
    """Test the update_E method."""
    state = mock_boundary.init_state()
    E = jnp.zeros((3, 3, 3))
    inverse_permittivity = jnp.ones((3, 3, 3))
    updated_E = mock_boundary.update_E(E, state, inverse_permittivity)
    assert (updated_E == E).all()


def test_update_H(mock_boundary: MockBoundary):
    """Test the update_H method."""
    state = mock_boundary.init_state()
    H = jnp.zeros((3, 3, 3))
    inverse_permeability = 1.0
    updated_H = mock_boundary.update_H(H, state, inverse_permeability)
    assert (updated_H == H).all()


def test_interface_grid_shape(mock_boundary: MockBoundary):
    """Test the interface_grid_shape method."""
    shape = mock_boundary.interface_grid_shape()
    assert shape == (1, 20, 30)


def test_interface_slice_tuple(mock_boundary: MockBoundary):
    """Test the interface_slice_tuple method."""
    slice_tuple = mock_boundary.interface_slice_tuple()
    assert slice_tuple == ((0, 1), (0, 20), (0, 30))


def test_interface_slice(mock_boundary: MockBoundary):
    """Test the interface_slice method."""
    slice_result = mock_boundary.interface_slice()
    assert slice_result == (slice(0, 1), slice(0, 20), slice(0, 30))

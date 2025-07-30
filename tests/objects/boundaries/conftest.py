import jax.numpy as jnp
import pytest

from fdtdx.typing import SliceTuple3D


class MockSimulationConfig:
    """Mock simulation configuration for testing purposes."""

    def __init__(self, dtype: jnp.dtype = jnp.float32, courant_number: float = 0.99):
        self.dtype = dtype
        self.courant_number = courant_number


@pytest.fixture
def mock_simulation_config():
    """Fixture to create a mock simulation configuration."""
    return MockSimulationConfig()


@pytest.fixture
def grid_slice_tuple() -> SliceTuple3D:
    """Fixture to provide a grid slice tuple for testing."""
    return ((0, 10), (0, 20), (0, 30))


@pytest.fixture
def grid_slice_tuple_small() -> SliceTuple3D:
    """Fixture to provide a grid slice tuple for testing."""
    # return ((0, 3), (1, 4), (2, 6))
    return ((0, 3), (0, 4), (0, 5))

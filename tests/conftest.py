import numpy as np
import pytest

from dacr.data import RoutingData
from dacr.synthetic import make_synthetic_long


@pytest.fixture(scope="session")
def synthetic_long():
    return make_synthetic_long(n_queries=600, seed=1)


@pytest.fixture(scope="session")
def synthetic_data(synthetic_long) -> RoutingData:
    return RoutingData.from_long(synthetic_long)


@pytest.fixture
def rng():
    return np.random.default_rng(0)

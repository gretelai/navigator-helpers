import pytest

from navigator_helpers import SyntheticDataGenerator


def test_something():
    # Dummy test to verify pytest setup.

    with pytest.raises(TypeError):
        # Expected to fail with TypeError, because positional args are missing.
        SyntheticDataGenerator()

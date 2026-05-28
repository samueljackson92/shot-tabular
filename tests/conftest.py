import pytest

from shot_tabular.main import TimeSettings


@pytest.fixture
def time_settings() -> TimeSettings:
    return TimeSettings(tmin=0.0, tmax=2.0, dt=0.5, method="linear")

import pytest

from elastra import config


@pytest.fixture(scope="session")
def cfg():
    return config.load()


@pytest.fixture(scope="session")
def needs_assets(cfg):
    """Skip a test that compiles a robot scene when the assets are not downloaded."""

    if not (config.assets_dir(cfg) / "protomotions" / "g1_holo_compat.xml").is_file():
        pytest.skip("third-party assets missing: run elastra-download-assets")

# type: ignore
""" unit tests for workexperience2024.utils module (no external requirements)"""

from pathlib import Path

import pytest

from workexperience2024.utils import about, get_package_root, get_project_root


@pytest.fixture()
def expected_project_root():
    return Path(__file__).parent.parent.parent


@pytest.fixture()
def expected_package_root():
    return Path(__file__).parent.parent


class TestUtils:
    """Test utils"""

    def test_project_root_path(self, expected_project_root):
        assert isinstance(get_project_root(), Path)
        assert expected_project_root == get_project_root()

    def test_package_root_path(self, expected_package_root):
        assert isinstance(get_package_root(), Path)
        assert expected_package_root == get_package_root()


@pytest.fixture()
def about_string():
    return about()


class TestAbout:
    """
    Test About function comprehensively; nothing is getting past this lot!
    """

    def test_about_returns_str(self, about_string):
        assert isinstance(about_string, str)

    def test_string_contains_name(self, about_string):
        name = get_package_root().name
        assert f"Project Name: {name}" in about_string

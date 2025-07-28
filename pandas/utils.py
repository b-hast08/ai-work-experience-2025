"""
General Utility functions for workexperience2024
"""

from pathlib import Path

import toml


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).resolve().parent.parent


def get_package_root() -> Path:
    """Returns package root folder."""
    return Path(__file__).resolve().parent


def about() -> str:
    """
    Standard greeter function to give the documents something to cover
    regarding the project. The project information is extracted from pyproject.toml.

    Returns:
        Information about the project and key contacts.

    Example:
        >>> about()
        "Project Name: PROJECT_NAME
        Project Summary: PROJECT_SUMMARY
        Author Name: AUTHOR_NAME
        Author Contact: AUTHOR_CONTACT
        Hartree Centre Data Science Team:
            HartreeCentreDataScience@stfc365.onmicrosoft.com
        Hartree Centre: https://www.hartree.stfc.ac.uk/Pages/home.aspx"
    """
    parent_folder = Path(__file__).resolve().parent.parent
    config_file = Path(parent_folder / "pyproject.toml")
    config = toml.load(config_file)
    return (
        f"Project Name: {config['tool']['poetry']['name']}\n"
        f"Project Description: {config['tool']['poetry']['description']}\n"
        f"Authors: {config['tool']['poetry']['authors']}\n"
        "Hartree Centre Data Science Team: "
        "HartreeCentreDataScience@stfc365.onmicrosoft.com\n"
        "Hartree Centre: https://www.hartree.stfc.ac.uk/Pages/home.aspx\n"
    )

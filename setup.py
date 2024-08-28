from pathlib import Path

from setuptools import find_packages, setup


def reqs(file_path: str) -> list[str]:
    with open(Path(file_path)) as fh:
        return [
            r.strip()
            for r in fh.readlines()
            if not (r.startswith("#") or r.startswith("\n"))
        ]

setup(
    name="navigator-helpers",
    version="0.1.2",  # Incremented version number
    packages=find_packages(exclude=["tests", "examples"], include=["navigator_helpers"]),
    python_requires=">=3.9",
    install_requires=reqs("requirements.txt"),
    extras_require={
        "dev": reqs("requirements-dev.txt"),
    },
    entry_points={"console_scripts": []},
    author="Gretel",
    author_email="hi@gretel.ai",
    description="A library of helper functions for Gretel Navigator, including synthetic data generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gretelai/navigator-helpers",
    license="https://gretel.ai/license/source-available-license",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Free To Use But Restricted",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
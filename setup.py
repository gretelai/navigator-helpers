from setuptools import setup, find_packages

setup(
    name="navigator-helpers",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        datasets==2.19.0
        gretel-client==0.19.0
        langchain==0.2.2
        pandas==2.2.1
        streamlit==1.35.0
        tqdm==4.66.4
    ],
    author="Gretel",
    author_email="hi@gretel.ai",
    description="A library of helper functions for Gretel Navigator",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gretelai/navigator-helpers",
    license="https://gretel.ai/license/source-available-license",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Free To Use But Restricted",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

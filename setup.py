from setuptools import setup, find_packages

setup(
    name="tech-pathway-connector",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "networkx",
        "matplotlib",
    ],
) 
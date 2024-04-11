import setuptools

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mattstools",
    version="1.0.0",
    author="Matthew Leigh",
    author_email="mattcleigh@gmail.com",
    description="Some common utilities used in my DL projects",
    long_description=long_description,
    url="https://gitlab.cern.ch/mleigh/mattstools",
    project_urls={"Bug Tracker": "https://gitlab.cern.ch/mleigh/mattstools/issues"},
    license="MIT",
    packages=["mattstools", "mattstools.flows", "mattstools.gnets"],
    install_requires=[
        "dotmap",
        # "EnergyFlow",
        "geomloss",
        "matplotlib",
        "nflows",
        "numpy",
        "pandas",
        "PyYAML",
        "scikit_learn",
        "scipy",
        "setuptools",
        "torch",
        "tqdm",
        "wandb",
    ],
)

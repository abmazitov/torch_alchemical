from setuptools import setup, find_packages


setup(
    name="torch_alchemical",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "ase",
        "torch",
        "scipy",
        "torch_geometric",
        "lightning",
        "torch_spex@git+https://github.com/frostedoyster/torch_spex.git@08cfe0d",
    ],
    dependency_links=["https://download.pytorch.org/whl/cpu"],
)

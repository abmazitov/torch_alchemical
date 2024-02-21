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
        "wandb",
        "ruamel.yaml",
        "torch_spex@git+https://github.com/lab-cosmo/torch_spex.git",
        "metatensor[torch]",
    ],
    dependency_links=["https://download.pytorch.org/whl/cpu"],
)

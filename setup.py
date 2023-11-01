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
        "torch_spex@git+https://github.com/lab-cosmo/torch_spex.git@5fb0d07",
        "metatensor[torch]@git+https://github.com/lab-cosmo/metatensor.git@ef15c73",
    ],
    dependency_links=["https://download.pytorch.org/whl/cpu"],
)

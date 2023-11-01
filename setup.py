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
        "torch_spex@git+https://github.com/lab-cosmo/torch_spex.git@80cf96b",
        "metatensor[torch]@git+https://github.com/lab-cosmo/metatensor/archive/0436e27.zip",
    ],
    dependency_links=["https://download.pytorch.org/whl/cpu"],
)

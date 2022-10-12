from setuptools import setup, find_packages

setup(
    name="edgar",
    version="1.0",
    author="Tobias Deusser, Syed Musharraf Ali, Lars Hillebrand",
    packages=find_packages(),
    install_requires=[
        "aim",
        "babel",
        "fluidml",  # install from source (branch run-info-access)
        "gensim",
        "jinja2",
        "lxml",
        "numpy",
        "openpyxl",
        "pandas",
        "pyyaml",
        "secedgar==0.4.0",
        "sklearn",
        "syntok",
        "tensorboard",
        "torch>=1.8.1",  # old: 1.7.0
        "tqdm",
        "transformers",
        "wandb",
        "xlrd==1.2.0",  # current version can not read in Excel anymore
    ],
    extras_require={"jupyter": ["jupyterlab", "jupyter_contrib_nbextensions"]},
)

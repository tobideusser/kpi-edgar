from setuptools import setup, find_packages

# todo: clean up the requirements, they are from the old KRE repo

setup(
    name="edgar",
    version="0.1",
    author="Tobias Deusser, Lars Hillebrand, Syed Musharraf Ali",
    packages=find_packages(),
    install_requires=["aim",
                      "babel",
                      "fluidml",
                      "jinja2",
                      "lxml",
                      "matplotlib",
                      "seaborn",
                      "numpy",
                      "openpyxl",
                      "pandas",
                      "pyyaml",
                      "rapidfuzz",
                      "secedgar",  # requires version 0.4 from github
                      "sklearn",
                      "syntok",
                      "tensorboard",
                      "termcolor",
                      "torch>=1.8.1",  # old: 1.7.0
                      "tqdm",
                      "transformers",
                      "unidecode",    # NOT MIT
                      "xlrd==1.2.0",  # current version can not read in excel anymore
                      "xlwt",
                      "wandb"],
    extras_require={"jupyter": ["jupyterlab",
                                "jupyter_contrib_nbextensions"]}
)

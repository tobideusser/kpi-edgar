from setuptools import setup, find_packages

setup(
    name='kpi_relation_extractor',
    version='0.1',
    author='Lars Hillebrand, Tobias Deusser',
    packages=find_packages(),
    install_requires=['aim',
                      'babel',
                      'fluidml',
                      'jinja2',
                      'matplotlib',
                      'seaborn',
                      'numpy',
                      'openpyxl',
                      'pandas',
                      'pyyaml',
                      'rapidfuzz',
                      'secedgar',
                      'sklearn',
                      'syntok',
                      'tensorboard',
                      'termcolor',
                      'torch>=1.8.1',  # old: 1.7.0
                      'tqdm',
                      'transformers',
                      'unidecode',    # NOT MIT
                      'xlrd==1.2.0',  # current version can not read in excel anymore
                      'xlwt',
                      'wandb'],
    extras_require={'jupyter': ['jupyterlab',
                                'jupyter_contrib_nbextensions']}
)

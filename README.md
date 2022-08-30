# How to run

This repository lets you run the download, training, and evaluation process for the paper "KPI-EDGAR: A Novel Dataset and Accompanying Metric for Relation Extraction from Financial Documents". 

First, install all required packages by running ***pip install /path/to/setup.py***, preferably in a virtual environment.

Then, download the data from the EDGAR database with the ***download_from_edgar.py*** located under the folder ***/scripts***.

Thereafter, kick of the training pipeline by adding the relevant folder to the config under the folder ***/config*** 
and executing the ***run_train_pipeline.py*** located under ***/scripts***.

# KPI-EDGAR dataset

If you simply want to download the KPI-EDGAR dataset, access the annotations and data from ***/data***. 
The most up-to-date dataset will always be called ***kpi_edgar.xlsx***.
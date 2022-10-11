# How to run

This repository lets you run the download, training, and evaluation process for the paper "KPI-EDGAR: A Novel Dataset 
and Accompanying Metric for Relation Extraction from Financial Documents", to be published in the proceedings of the 
IEEE International Conference on Machine Learning and Applications 2022. 

First, install all required packages by running ***pip install /path/to/setup.py***, preferably in a virtual environment.

Then, download the data from the EDGAR database with the ***download_from_edgar.py*** located under the folder ***/scripts***.

Thereafter, kick of the training pipeline by adding the relevant folder to the config under the folder ***/config*** 
and executing the ***run_train_pipeline.py*** located under ***/scripts***.

# KPI-EDGAR dataset

If you simply want to download the KPI-EDGAR dataset, access the annotations and data from ***/data***. 
The most up-to-date dataset will always be called ***kpi_edgar.xlsx***.

# Citation

If you use KPI-EDGAR in academic work, please cite it directly:

```
@inproceedings{deusser2022kpi-edgar,
  title={KPI-EDGAR: A Novel Dataset and Accompanying Metric for Relation Extraction from Financial Documents},
  author={Deu{\ss}er, Tobias and Ali, Syed Musharraf and Nurchalifah, Desiana and Jacob, Basil and Bauckhage, Christian and Sifa, Rafet},
  booktitle={Proc. ICMLA},
  year={2022}
}
```
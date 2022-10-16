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

# Test set performances

We are maintaining a table with results on our test set. If you want your model listed here, simply send us your results 
and how you want to be cited.

| Model                 | Relation F<sub>1</sub> Score in % | Adjusted Relation F<sub>1</sub> Score in % |
|:----------------------|:---------------------------------:|:------------------------------------------:|
| KPI-BERT<sup>1</sup>  |               22.68               |                   43.76                    |
| SpERT<sup>1</sup>     |               20.95               |                   40.04                    |
| EDGAR–W2V<sup>1</sup> |               6.13                |                   19.71                    |
| GloVe<sup>1</sup>     |               5.11                |                   17.18                    |

<sup>1</sup>Baseline introduced in "KPI-EDGAR: A Novel Dataset 
and Accompanying Metric for Relation Extraction from Financial Documents".

# Citation

If you use KPI-EDGAR in academic work, please cite it directly:

```
@inproceedings{kpi-edgar,
  title={KPI-EDGAR: A Novel Dataset and Accompanying Metric for Relation Extraction from Financial Documents},
  author={Deu{\ss}er, Tobias and Ali, Syed Musharraf, and Hillebrand, Lars and Nurchalifah, Desiana and Jacob, Basil and Bauckhage, Christian and Sifa, Rafet},
  booktitle={Proc. ICMLA (to be published)},
  year={2022}
}
```

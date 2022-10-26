# How to run

This repository lets you run the download, training, and evaluation process for the paper "KPI-EDGAR: A Novel Dataset 
and Accompanying Metric for Relation Extraction from Financial Documents", to be published in the proceedings of the 
IEEE International Conference on Machine Learning and Applications 2022. A preprint of the paper is available on 
[arxiv](https://arxiv.org/abs/2210.09163).

First, install all required packages by running ***pip install /path/to/setup.py***, preferably in a virtual 
environment. NOTE: We used the most up-to-date branch of [fluidml](https://github.com/fluidml/fluidml) for this project,
so you likely have to install the branch "run-info-access" from source.

Then, download the data from the EDGAR database with the ***download_from_edgar.py*** located under the folder ***/scripts***.

Thereafter, kick of the training pipeline by adding the relevant folder to the config under the folder ***/config*** 
and executing the ***run_train_pipeline.py*** located under ***/scripts***.

# KPI-EDGAR dataset

If you simply want to download the KPI-EDGAR dataset, access the annotations and data from ***/data***. 
The most up-to-date dataset will always be called ***kpi_edgar.xlsx***.

In ***/data***, there is also a "pre-parsed" json file titled ***kpi_edgar.json***, which includes IOBES tags and might 
be easier to use for some.

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

If you use KPI-EDGAR in your academic work, please cite it directly:

```
@inproceedings{kpi-edgar,
  title={KPI-EDGAR: A Novel Dataset and Accompanying Metric for Relation Extraction from Financial Documents},
  author={Deu{\ss}er, Tobias and Ali, Syed Musharraf and Hillebrand, Lars and Nurchalifah, Desiana and Jacob, Basil and Bauckhage, Christian and Sifa, Rafet},
  booktitle={Proc. ICMLA},
  year={2022},
  doi={10.48550/arXiv.2210.09163}
}
```

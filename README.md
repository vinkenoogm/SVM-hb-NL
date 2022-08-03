# SVM-hb-NL

[![Python Version](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![DOI](https://zenodo.org/badge/511832055.svg)][Zenodo]
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vinkenoogm/SVM-hb-NL/HEAD)

This repository contains the code used to obtain the results that are described
in the scientific article "Explainable hemoglobin deferral predictions using
machine learning models: interpretation and consequences for the blood supply"
(manuscript currently under review). The repository has been indexed on [Zenodo].


## Data 
The data used in this analysis is collected by Sanquin Blood Supply Foundation,
from donors who have given permission for the use of their data in scientific
research. Due to privacy reasons this data will not be shared. The /data folder
contains a description of the raw data, so that researchers with access to
similar data may use this code to analyse their own data.

## Scientific abstract
> Accurate hemoglobin (Hb) deferral predictions for whole-blood donors could aid
> blood banks in reducing deferral rates and increasing efficiency and donor
> motivation. Complex models are needed to make accurate predictions, but
> predictions must also be explainable. Before implementation of a prediction
> model, its impact on the blood supply should be estimated to avoid shortages.
> Donation visits between October 2017 and December 2021 were selected from
> Sanquin’s database system. The following variables were available for each
> visit: donor sex, age, donation start time, month, number of donations in the
> last 24 months, most recent ferritin level, days since last ferritin
> measurement, Hb at nth previous visit (n between 1 and 5), days since nth
> previous visit. Outcome variable Hb deferral has two classes: deferred and not
> deferred. Support vector machines were used as prediction models, and SHapley
> Additive exPlanations (SHAP) values were used to quantify the contribution of
> each variable to the model predictions. Performance was assessed using
> precision and recall of both outcome classes. The potential impact on blood
> supply was estimated by predicting deferral at earlier or later donation dates.
> We present a model that predicts Hb deferral in an explainable way. If used in
> practice, 64% of non-deferred donors would be invited on or before their
> original donation date, while 80% of deferred donors would be invited later.
> With a shorter average donation interval, the number of blood bank visits
> would increase by 7%, while deferral rates would decrease by 60% (currently
> 3% for women, 1% for men).

## Requirements
The file `requirements.txt` contains all necessary (Python) packages along with
version information. All code was run using Python 3.10.4. With large datasets,
code in .py files is computationally expensive to run and running on a HPC or
similar is recommended.

## Models
Five different models (SVM-1 through SVM-5) are trained separately for men and
women, resulting in ten models total. The number in the model name indicates how
many previous Hb measurements are used in the prediction. As donors can only be
included in SVM-n if they have at least n previous visits, sample sizes decrease
from SVM-1 to SVM-5. The following predictor variables are used:

Variable	 | Unit or values |	Description
-------------|----------------|----------------------------------------------------------------------------------------------
Sex	         | {male, female} |	Biological sex of the donor; separate models are trained for men and women
Age          | years          |	Donor age at time of donation
Time         | hours          |	Registration time when the donor arrived at the blood bank
Month        | {1-12}         |	Month of the year that the visit took place
NumDon       | count          |	Number of successful (collected volume > 250 mL) whole-blood donations in the last 24 months
FerritinPrev | ng/mL          |	Most recent ferritin level measured in this donor
DaysSinceFer | days           |	Time since this donor’s last ferritin measurement
HbPrevn      | mmol/L         |	Hemoglobin level at nth previous visit, for n between 1-5
DaysSinceHbn | days	          | Time since related Hb measurement at nth previous visit, for n between 1-5


## Files
Code files are numbered in order of use. Python scripts all use the following arguments:

Argument     | Description
-------------|--------------------------------------------------------
nback        | [int] Which model to use (number of previous donations)
sex          | [men/women] Use male or female donors
foldersuffix | [str] Optional foldersuffix to specify a run 

For example:

```
$ python src/1_hyperparams.py 3 men
```

To see which arguments are accepted, you can use `--help`:

```
$ python src/1_hyperparams.py --help
usage: 1_hyperparams.py [-h] [--foldersuffix FOLDERSUFFIX] nback {men,women}

positional arguments:
  nback                 [int] number of previous Hb values to use in prediction
  {men,women}           [men/women] sex to use in model

options:
  -h, --help            show this help message and exit
  --foldersuffix FOLDERSUFFIX
                        [str] optional suffix indicating non-default run
```

### 0_preprocessing.ipynb
This notebook takes the raw donation data (source files) as collected by Sanquin.
Preprocessing includes
- merging donation files from different years
- selecting relevant donations and variables
- manipulating recorded variables into required predictor variables
- scaling the variables to N(0,1)
- saving train and test data sets
- describing marginal distributions of predictor variables.

### 1_hyperparams.py
Run with arguments: 
This script takes as input the scaled train data sets produced in
`0_preprocessing.ipynb`. Using a grid search with 5-fold cross-validation,
hyperparameters C and gamma are optimized for support vector machines with RBF
kernel. The results of the grid search are saved in `results/hyperparams/`.

### 2_modeltraining.py
In this script all models are trained using the scaled train data sets and the
optimized hyperparameters. Performance (precision and recall on both outcome
classes) is assessed on both the train and test sets. Trained models and
performance metrics are saved in `results/models/` (`clf\_{sex}\_{n}.sav` are
trained models files, `res\_{sex}\_{n}.pkl` contain performance metrics).

### 3_modelperformance.ipynb
This notebook reads the `res\_{sex}\_{n}.pkl` files and creates graphs that show
model performance. These plots are also saved in `results/plots_performance/`.

### 4_calcshap.py
Additional argument:

Argument     | Description
-------------|---------------------------------------------------------------------
n            | [int] Number of randomly selected donors to calculate SHAP values on

This script calculates SHAP values for a random subset of donors from the test
set. Results are not shared because they contain donor-level sensitive information.

### 5_plotshap.ipynb
In this notebook, summary plots for the SHAP values calculated by
`4_calcshap.py` are created and saved in `results/plots_shap/`.

### 6_changingtime.py
This script uses the models trained in `2_modeltraining.py` and manipulates
time-related predictor variables to make predictions for different invitation
dates.

### 7_impactbloodsupply.ipynb
This notebook uses the prediction results from `6_changingtime.py` to assess
impact on the blood supply, should these models be used to guide donor
invitations. It saves a version of the prediction files without sensitive donor
information in `results/pred_timechange/` and plots are saved to
`results/plots_performance/`.

[Zenodo]: https://zenodo.org/badge/latestdoi/511832055

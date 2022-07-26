import datetime
from pathlib import Path
import pickle
from pyprojroot import here
import sys
import warnings

import pandas as pd
import shap

warnings.filterwarnings('ignore')
data_path = here('data/')
results_path = here('results/')

sex = sys.argv[1]
nback = sys.argv[2]
foldersuffix = sys.argv[3]
n = sys.argv[4]


def calc_shap(nback, sex, n=100):
    filename = results_path / f'models{foldersuffix}/clf_{nback}.sav'
    clf = pickle.load(open(filename, 'rb'))

    index = 0 if sex == 'men' else 1
    val = pd.read_pickle(data_path / f'scaled{foldersuffix}/{sex}_{nback}_test.pkl')
    clf_s = clf[index]
    X_val = val[val.columns[:-1]]
    X_shap = shap.sample(X_val, n)
    explainer = shap.KernelExplainer(clf_s.predict, X_shap)
    shapvals = explainer.shap_values(X_shap)

    output_path = results_path / f'shap{foldersuffix}/'
    output_path.mkdir(parents=True, exist_ok=True)
    filename1 = f'Xshap_{sex}_{nback}_{n}.pkl'
    filename2 = f'shapvals_{sex}_{nback}_{n}.pkl'

    pickle.dump(X_shap, open(output_path / filename1, 'wb'))
    pickle.dump(shapvals, open(output_path / filename2, 'wb'))


print(f'Calculating shap values for {sex} nback {nback} (N = {n}) starting at {datetime.datetime.now()}')
calc_shap(nback, sex, int(n))

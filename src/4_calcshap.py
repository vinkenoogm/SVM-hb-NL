import datetime
import pandas as pd
from pathlib import Path
import pickle
import shap
import sys

import warnings 
warnings.filterwarnings('ignore')

data_path = Path('/data1/vinkenoogm/')
results_path = Path('/home/vinkenoogm/SVM-NL/results/')

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

    path = results_path / f'shap{foldersuffix}/'
    filename1 = f'Xshap_{sex}_{nback}_{n}.pkl'
    filename2 = f'shapvals_{sex}_{nback}_{n}.pkl'

    pickle.dump(X_shap, open(path+filename1, 'wb'))
    pickle.dump(shapvals, open(path+filename2, 'wb'))


print(f'Calculating shap values for  {sex}  nback {nback} ( N = {n} ) starting at {datetime.datetime.now()}')
calc_shap(nback, sex, int(n))

import argparse
import pickle
from pyprojroot import here
import warnings

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')
data_path = here('data/')
results_path = here('results/')

parser = argparse.ArgumentParser()
parser.add_argument('nback', type=int,
                    help="[int] ...")
parser.add_argument('sex', type=str, choices=['men', 'women'],
                    help="...")
parser.add_argument('foldersuffix', type=str,
                    help="...")
args = parser.parse_args()

nback, sex, foldersuffix = args.nback, args.sex, args.foldersuffix

print(sex)


train = pd.read_pickle(data_path / f'scaled{foldersuffix}/{sex}_{nback}_train.pkl')
X = train[train.columns[:-1]]
y = train['HbOK']

params = {'C': [10, 1, 0.1],
          'gamma': [1, 0.1, 0.01, 0.001],
          'kernel': ['rbf']}

gridsearch = GridSearchCV(estimator=SVC(class_weight='balanced'),
                          param_grid=params,
                          scoring='balanced_accuracy',
                          error_score='raise',
                          cv=5,
                          verbose=2)
gridsearch.fit(X, y)

output_folder = results_path / f'hyperparams{foldersuffix}/'
output_folder.mkdir(parents=True, exist_ok=True)  # create folder if it does not yet exist

filename = f'hyperparams_{sex}_{nback}.pkl'
with open(filename, 'wb') as handle:
    pickle.dump(gridsearch.cv_results_, handle, protocol=pickle.HIGHEST_PROTOCOL)

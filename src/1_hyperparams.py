import pandas as pd
from pathlib import Path
import pickle
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


data_path = Path('/data1/vinkenoogm/')
results_path = Path('../results/')

nback = int(sys.argv[1])
sex = sys.argv[2]
foldersuffix = sys.argv[3]

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

filename = results_path / f'hyperparams{foldersuffix}/hyperparams_{sex}_{nback}.pkl'
with open(filename, 'wb') as handle:
    pickle.dump(gridsearch.cv_results_, handle, protocol=pickle.HIGHEST_PROTOCOL)

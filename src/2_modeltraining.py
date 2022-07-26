import datetime
import pandas as pd
from pathlib import Path
import pickle
import sys
from sklearn.metrics import classification_report
from sklearn.svm import SVC

data_path = Path('/data1/vinkenoogm/')
results_path = Path('/home/vinkenoogm/SVM-NL/results/')

foldersuffix = sys.argv[1]
nbacks = sys.argv[2:]


def train_svm(data, hyperparams):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]
    
    hyp_c = hyperparams['C']
    hyp_g = hyperparams['gamma']
    hyp_k = hyperparams['kernel']
    
    clf = SVC(C=hyp_c, gamma=hyp_g, kernel=hyp_k, probability=True, class_weight='balanced')
    clf.fit(X, y.values.ravel())
    
    return(clf)

def calc_accuracy(clf, data):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]
    
    y_pred = clf.predict(X)
    
    return(classification_report(y, y_pred, output_dict=True))

def do_svm(nback):
    results = []
    clfs = []
    for sex in ['men', 'women']:
        print(f'Sex: {sex}  -  {datetime.datetime.now()}')
        train = pd.read_pickle(data_path / f'scaled{foldersuffix}/{sex}_{nback}_train.pkl')
        test = pd.read_pickle(data_path / f'scaled{foldersuffix}/{sex}_{nback}_test.pkl')

        hyps_all = pd.read_pickle(results_path / f'hyperparams{foldersuffix}/hyperparams_{sex}_{nback}.pkl')
        hyps_all = pd.DataFrame.from_dict(hyps_all)
        hyps = hyps_all.loc[hyps_all.rank_test_score == 1, 'params']
        hyps = hyps[hyps.index[0]]
        
        print('  Training SVM - ', datetime.datetime.now())
        clf = train_svm(train, hyps)
        
        print('  Calculating accuracy - ', datetime.datetime.now())
        cl_rep_train = calc_accuracy(clf, train)
        cl_rep_val = calc_accuracy(clf, test)
        results.extend([cl_rep_train, cl_rep_val])
        clfs.append(clf)
    return(results, clfs)

for nback in nbacks:
    res, clf = do_svm(int(nback))
    filename1 = results_path / f'models{foldersuffix}/res_{nback}.pkl'
    filename2 = results_path / f'models{foldersuffix}/clf_{nback}.sav'
    pickle.dump(res, open(filename1, 'wb'))
    pickle.dump(clf, open(filename2, 'wb'))

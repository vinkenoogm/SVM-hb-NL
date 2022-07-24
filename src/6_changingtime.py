import pandas as pd
from pathlib import Path
import pickle
import sys

data_path = Path('/data1/vinkenoogm/')
result_path = Path('/home/vinkenoogm/SVM-NL/results')

sex = sys.argv[1]
nback = int(sys.argv[2])
foldersuffix = sys.argv[3]
print(sex, nback, foldersuffix)


def make_preds(data, clf):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]
    y_pred = clf.predict(X)
    return(y_pred)


data = pd.read_pickle(data_path / f'scaled{foldersuffix}/{sex}_{nback}_test.pkl')
clf = pickle.load(open(result_path / f'models{foldersuffix}/clf_{nback}.sav', 'rb'))
clf_s = clf[0] if sex == 'men' else clf[1]
scaler = pickle.load(open(result_path / f'scalers{foldersuffix}/{sex}_{nback}.pkl', 'rb'))

print('loaded data')

data_res = data.copy()
y_pred_first = make_preds(data, clf_s)
data_res['HbOK_pred'] = y_pred_first

timecols = ['TimetoFer', *[f'TimetoPrev{str(n)}' for n in range(1, nback+1)]]
for timestep in range(-364, 371, 7):
    print(timestep)
    data_timestep = data.copy()
    data_timestep[data_timestep.columns[:-1]] = scaler.inverse_transform(data_timestep[data_timestep.columns[:-1]])
    data_timestep[timecols] = data_timestep[timecols].add(timestep)
    data_timestep['Month'] = (data_timestep['Month'] - 1 + round(timestep/30)) % 12 + 1
    data_timestep['Age'] = data_timestep['Age'] + (timestep / 365)
    data_timestep[data_timestep.columns[:-1]] = scaler.transform(data_timestep[data_timestep.columns[:-1]])

    y_pred = make_preds(data_timestep, clf_s)

    varname = f'HbOK_pred_{timestep}'
    data_res[varname] = y_pred
    if timestep == 0:
        data_res = data_res.copy()

print(data_res.head())
path = data_path / f'pred_timechange{foldersuffix}/'
data_res.to_pickle(path / f'data_res_{sex}_{nback}.pkl')

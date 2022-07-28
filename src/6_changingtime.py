import argparse
import pickle
from pyprojroot import here

import pandas as pd


data_path = here('data/')
result_path = here('results/')

parser = argparse.ArgumentParser()
parser.add_argument('nback', type=int,
                    help='[int] number of previous Hb values to use in prediction')
parser.add_argument('sex', type=str, choices=['men', 'women'],
                    help='[men/women] sex to use in model')
parser.add_argument('--foldersuffix', type=str, default='',
                    help='[str] optional suffix indicating non-default run')
args = parser.parse_args()

sex, nback, foldersuffix = args.sex, args.nback, args.foldersuffix
print(sex, nback, foldersuffix)


def make_preds(data, clf):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]
    return clf.predict(X)


data = pd.read_pickle(data_path / f'scaled{foldersuffix}/{sex}_{nback}_test.pkl')
clf = pickle.load(open(result_path / f'models{foldersuffix}/clf_{sex}_{nback}.sav', 'rb'))
scaler = pickle.load(open(result_path / f'scalers{foldersuffix}/{sex}_{nback}.pkl', 'rb'))

print('loaded data')

data_res = data.copy()
y_pred_first = make_preds(data, clf)
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

    y_pred = make_preds(data_timestep, clf)

    varname = f'HbOK_pred_{timestep}'
    data_res[varname] = y_pred
    if timestep == 0:
        data_res = data_res.copy()

print(data_res.head())
output_path = result_path / f'pred_timechange{foldersuffix}/'
output_path.mkdir(parents=True, exist_ok=True)
data_res.to_pickle(output_path / f'data_res_{sex}_{nback}.pkl')

import unittest

import pandas as pd
from sklearn.model_selection import KFold

from lightgbm_utils import run_lightgbm_model, DEFAULT_PARAMETERS


class LightGbmUtils(unittest.TestCase):

    def test_run_lightgbm_model(self):
        df_train = pd.read_csv('./data/application_train.csv')
        df_test = pd.read_csv('./data/application_test.csv')
        y = df_train['TARGET']
        del df_train['TARGET']
        feats = [f for f in df_train.columns if f not in ['SK_ID_CURR']]
        folds = KFold(n_splits=2, shuffle=True, random_state=42)
        args = DEFAULT_PARAMETERS.copy()
        df_fold_preds_train, df_fold_preds_test, df_feature_importance = \
            run_lightgbm_model(df_train, df_test, y, folds=folds, feats=feats, early_stopping=10, save_model=False,
                               args_dict=args)

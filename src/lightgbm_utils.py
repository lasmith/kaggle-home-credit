import gc

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier, Booster

# Default parameters for LightGBM. Some of these are the defaults, some are custom. The ones listed below are
# largely the ones recommended in parameter tuning in the docs
DEFAULT_PARAMETERS = {
    "n_estimators": 4000,
    "learning_rate": 0.03,
    "num_leaves": 30,
    "colsample_bytree": 0.8,
    "subsample": 0.9,
    "max_depth": 6,
    "max_bin": 255,
    "num_iterations": 100,
    "min_data_in_leaf": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_split_gain": 0.01,
    "min_child_weight": 2,
    "silent": -1,
    "verbose": -1,
    "objective": "regression",
    "metric": "",
    "bagging_fraction": 1.0,
    "bagging_freq": 0.0,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "min_gain_to_split": 0.0,
    "feature_fraction": 1.0

}



def run_lightgbm_model(df_train, df_test, df_target, folds, feats, early_stopping, args_dict,
                       save_model=False, file_prefix=None,
                       ):
    # Somewhere to store the results from the cross folds
    df_fold_preds_train = np.zeros(df_train.shape[0])
    df_fold_preds_test = np.zeros(df_test.shape[0])
    df_feature_importance = pd.DataFrame()
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(df_train)):
        trn_x, trn_y = df_train[feats].iloc[trn_idx], df_target.iloc[trn_idx]
        val_x, val_y = df_train[feats].iloc[val_idx], df_target.iloc[val_idx]

        clf = LGBMClassifier(**args_dict)
        clf.fit(trn_x, trn_y, eval_set=[(trn_x, trn_y), (val_x, val_y)], eval_metric='[l1,l2]',
                verbose=100, early_stopping_rounds=early_stopping)

        if save_model:
            booster: Booster = clf.booster_
            booster.save_model(file_prefix + "_fold" + str(n_fold) + ".txt")

        df_fold_preds_train[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        df_fold_preds_test += clf.predict_proba(df_test[feats], num_iteration=clf.best_iteration_)[:,
                              1] / folds.n_splits

        df_fold_features = pd.DataFrame()
        df_fold_features["feature"] = feats
        df_fold_features["importance"] = clf.feature_importances_
        df_fold_features["fold"] = n_fold + 1
        df_feature_importance = pd.concat([df_feature_importance, df_fold_features], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, df_fold_preds_train[val_idx])))

        # Free up some memory
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Overall AUC score %.6f' % roc_auc_score(df_target, df_fold_preds_train))
    return df_fold_preds_train, df_fold_preds_test, df_feature_importance

import gc

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier, Booster


def create_model(learning_rate=0.03, num_leaves=30, max_depth=6, min_data_in_leaf=20, objective='regression',
                 metric=''):
    return LGBMClassifier(
        n_estimators=4000,
        learning_rate=learning_rate,
        num_leaves=num_leaves,  # somewhere around / below 2^max_depth
        colsample_bytree=.8,
        subsample=.9,
        max_depth=max_depth,
        min_data_in_leaf=min_data_in_leaf,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=2,
        silent=-1,
        verbose=-1,
        objective=objective,
        metric=metric
    )


# TODO: Create param object
def run_lightgbm_model(df_train, df_test, df_target, folds, feats, early_stopping, save_model=False, file_prefix=None,
              learning_rate=0.03, num_leaves=30, max_depth=6, min_data_in_leaf=20, objective='regression', metric=''):
    # Somewhere to store the results from the cross folds
    df_fold_preds_train = np.zeros(df_train.shape[0])
    df_fold_preds_test = np.zeros(df_test.shape[0])
    df_feature_importance = pd.DataFrame()
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(df_train)):
        trn_x, trn_y = df_train[feats].iloc[trn_idx], df_target.iloc[trn_idx]
        val_x, val_y = df_train[feats].iloc[val_idx], df_target.iloc[val_idx]

        clf = create_model(learning_rate, num_leaves, max_depth, min_data_in_leaf, objective, metric)
        clf.fit(trn_x, trn_y, eval_set=[(trn_x, trn_y), (val_x, val_y)], eval_metric='auc',
                verbose=100, early_stopping_rounds=early_stopping)

        if save_model:
            booster: Booster = clf.booster_
            booster.save_model(file_prefix+"_fold"+str(n_fold)+".txt")

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

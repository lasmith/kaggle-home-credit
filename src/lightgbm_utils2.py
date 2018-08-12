import lightgbm as lgb


def save_data_set(df, file_name):
    data = lgb.Dataset(df)
    data.save_binary(file_name)


DEFAULT_PARAMETERS = {
    "n_estimators": 4000,
    "learning_rate": 0.03,
    "num_leaves": 30,
    "colsample_bytree": 0.8,
    "subsample": 0.9,
    "max_depth": 6,
    "min_data_in_leaf": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_split_gain": 0.01,
    "min_child_weight": 2,
    "silent": -1,
    "verbose": -1,
    "objective": "regression",
    "metric": ""
}


def create_model(df_train, df_test, y, num_round, early_stopping_rounds, params=DEFAULT_PARAMETERS, no_folds=None):
    """

    :param df_train: The training data frame
    :param df_test: The testing data frame
    :param y: The labels for the training data set
    :param params: A Json string of the parameters
    :return:
    """
    train_data = lgb.Dataset(df_train, label=y)
    test_data = lgb.Dataset(df_test)
    if no_folds:
        bst = lgb.cv(params, train_data, num_round, nfold=no_folds)
    else:
        bst = lgb.train(params, train_data, num_round, valid_sets=[test_data],
                        early_stopping_rounds=early_stopping_rounds)

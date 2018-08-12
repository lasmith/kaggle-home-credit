import logging

import pandas as pd
import numpy as np
import gc

from parso.python import prefix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
    
def load_bureau_data(in_dir):
    """
    Load the bureau balance data
    :param in_dir:  The data dir
    :return: a dataframe with the engineered bureau balance information
    """
    
    buro_bal = pd.read_csv(in_dir + '/bureau_balance.csv')
    logger.debug("Bureau bal shape : %s" % str(buro_bal.shape))

    logger.debug('One hot encoding bureau balances status')
    buro_bal = pd.concat([buro_bal, pd.get_dummies(buro_bal.STATUS, prefix='buro_bal_status')], axis=1).drop('STATUS',
                                                                                                             axis=1)

    logger.debug('Counting bureau balances')
    buro_counts = buro_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
    buro_bal['buro_count'] = buro_bal['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])

    logger.debug('averaging buro bal')
    avg_buro_bal = buro_bal.groupby('SK_ID_BUREAU').mean()

    avg_buro_bal.columns = ['avg_buro_' + f_ for f_ in avg_buro_bal.columns]
    del buro_bal
    gc.collect()

    buro = pd.read_csv(in_dir + '/bureau.csv')

    logger.debug('Go to dummies')
    buro_credit_active_dum = pd.get_dummies(buro.CREDIT_ACTIVE, prefix='ca_')
    buro_credit_currency_dum = pd.get_dummies(buro.CREDIT_CURRENCY, prefix='cu_')
    buro_credit_type_dum = pd.get_dummies(buro.CREDIT_TYPE, prefix='ty_')

    buro_full = pd.concat([buro, buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum], axis=1)
    # buro_full.columns = ['buro_' + f_ for f_ in buro_full.columns]

    del buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum
    gc.collect()

    logger.debug('Merge with bureau avg')
    buro_full = buro_full.merge(right=avg_buro_bal.reset_index(), how='left', on='SK_ID_BUREAU',
                                suffixes=('', '_bur_bal'))

    logger.debug('Counting buro per SK_ID_CURR')
    nb_bureau_per_curr = buro_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
    buro_full['SK_ID_BUREAU'] = buro_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])

    logger.debug('Averaging bureau')
    avg_buro = buro_full.groupby('SK_ID_CURR').mean()
    logger.debug(avg_buro.head())

    del buro, buro_full
    gc.collect()

    return avg_buro.fillna(0)

def load_previous_applications(in_dir):
    """
    Load the previous application data and average i
    :param in_dir:
    :return:
    """
    logger.debug('Loading previous applications')
    prev = pd.read_csv(in_dir + '/previous_application.csv')
    prev_cat_features = [
        f_ for f_ in prev.columns if prev[f_].dtype == 'object'
    ]
    logger.debug('One hot encoding categorical features..')
    prev_dum = pd.DataFrame()
    for f_ in prev_cat_features:
        prev_dum = pd.concat([prev_dum, pd.get_dummies(prev[f_], prefix=f_).astype(np.uint8)], axis=1)
    prev = pd.concat([prev, prev_dum], axis=1)
    del prev_dum
    gc.collect()
    logger.debug('Counting number of Prevs')
    nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])
    logger.debug('Averaging prev')
    avg_prev = prev.groupby('SK_ID_CURR').mean()
    logger.debug(avg_prev.head())
    del prev
    gc.collect()
    return avg_prev.fillna(0)


def load_installments_data(in_dir):
    logger.debug('Reading Installments')
    inst = pd.read_csv(in_dir + '/installments_payments.csv')
    nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    avg_inst = inst.groupby('SK_ID_CURR').mean()
    avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]
    return avg_inst.fillna(0)


def load_credit_card_data(in_dir):
    logger.debug('Reading CC balance')
    cc_bal = pd.read_csv(in_dir + '/credit_card_balance.csv')
    logger.debug('One hot encoding contract statuses')
    cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'], prefix='cc_bal_status_')], axis=1)
    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    logger.debug('Compute average')
    avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
    avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]
    del cc_bal, nb_prevs
    gc.collect()
    return avg_cc_bal.fillna(0)


def load_pos_data(in_dir):
    logger.debug('Reading POS_CASH')
    pos = pd.read_csv(in_dir + '/POS_CASH_balance.csv')
    logger.debug('Go to dummies')
    pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'])], axis=1)
    logger.debug('Compute nb of prevs per curr')
    nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    logger.debug('Go to averages')
    avg_pos = pos.groupby('SK_ID_CURR').mean()
    del pos, nb_prevs
    gc.collect()
    return avg_pos.fillna(0)

def load_train_test_data_factorize(in_dir):
    logger.debug('Read data and test')
    data = pd.read_csv(in_dir + '/application_train.csv')
    test = pd.read_csv(in_dir + '/application_test.csv')
    logger.debug('Data Shape: %s, Test Shape: %s' % (data.shape, test.shape))
    y = data['TARGET']
    del data['TARGET']
    categorical_feats = [
        f for f in data.columns if data[f].dtype == 'object'
    ]
    logger.debug(categorical_feats)
    for f_ in categorical_feats:
        data[f_], indexer = pd.factorize(data[f_])
        test[f_] = indexer.get_indexer(test[f_])
    return data, test, y

def fix_missing_cols(in_train, in_test):
    missing_cols = set( in_train.columns ) - set( in_test.columns )
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        in_test[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    in_test = in_test[in_train.columns]
    return in_test

def load_train_test_data(in_dir, in_cols=None):
    logger.debug('Reading train and test data')
    df_train_pre = pd.read_csv(in_dir + '/application_train.csv')
    df_test_pre = pd.read_csv(in_dir + '/application_test.csv')
    logger.debug('Data Shape Pre-filter: %s, Test Shape: %s' % (df_train_pre.shape, df_test_pre.shape))
    y = df_train_pre['TARGET']
    del df_train_pre['TARGET']
    if in_cols:
        df_train = df_train_pre[in_cols]
        df_test = df_test_pre[in_cols]
    else:
        df_train = df_train_pre
        df_test = df_test_pre
    logger.debug('Data Shape: %s, Test Shape: %s' % (df_train.shape, df_test.shape))
    logger.debug("Loaded training and test data")
    return df_train, df_test, y

def load_data_dummies(in_df_train, in_df_test):
    df_train = in_df_train
    df_test = in_df_test
    categorical_feats = [
        f for f in df_train.columns if df_train[f].dtype == 'object'
    ]
    logger.debug(categorical_feats)
    for f_ in categorical_feats:
        prefix=f_
        df_train = pd.concat([df_train, pd.get_dummies(df_train[f_], prefix=prefix)], axis=1).drop(f_, axis=1)
        df_test = pd.concat([df_test, pd.get_dummies(df_test[f_], prefix=prefix)], axis=1).drop(f_, axis=1)
        df_test = fix_missing_cols(df_train, df_test)
    return df_train, df_test

def normalize_numericals(df, cols):
    for c in cols:
        df[c] = df[c].fillna(0)
    df_to_norm = df[cols]
    df_arr = preprocessing.normalize(df_to_norm)
    for c in cols:
        del df[c]
    df_to_norm = pd.DataFrame(df_arr, columns=cols)
    return pd.concat([df, df_to_norm], sort=False, axis=1)


def append_data(df_test, df_train, in_dir, func):
    df_additional = func(in_dir)
    df_train = df_train.merge(right=df_additional.reset_index(), how='left', on='SK_ID_CURR')
    df_test = df_test.merge(right=df_additional.reset_index(), how='left', on='SK_ID_CURR')
    return df_train, df_test


def append_bureau_data(in_dir, df_train, df_test):
    return append_data(df_test, df_train, in_dir, load_bureau_data)

def append_previous_applications(in_dir, df_train, df_test):
    return append_data(df_test, df_train, in_dir, load_previous_applications)

def append_pos_data(in_dir, df_train, df_test):
    return append_data(df_test, df_train, in_dir, load_pos_data)

def append_credit_card_data(in_dir, df_train, df_test):
    return append_data(df_test, df_train, in_dir, load_credit_card_data)

def append_installments_data(in_dir, df_train, df_test):
    return append_data(df_test, df_train, in_dir, load_installments_data)

def load_model(in_dir):
    """
    Load the data and run a bit of feature engineering to create the data to train on. This will
    1. Create avg balances
    2. Count no. of previous credits
    3. Average credit card balances

    :param in_dir: The directory where the data is
    :return: data - the data frame for training,
             test - The data frame to test against
             y - The target to predict
    """
    avg_buro = load_bureau_data(in_dir)

    avg_prev = load_previous_applications(in_dir)

    avg_pos = load_pos_data(in_dir)

    avg_cc_bal = load_credit_card_data(in_dir)

    avg_inst = load_installments_data(in_dir)

    data, test, y = load_train_test_data_factorize(in_dir)

    data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

    data = data.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

    del avg_buro, avg_prev, avg_pos, avg_cc_bal, avg_inst
    gc.collect()

    return data, test, y


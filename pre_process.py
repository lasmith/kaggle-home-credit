import logging

import pandas as pd
import numpy as np
import gc

logger = logging.getLogger(__name__)
    
def load_bureau_data(in_dir):
    """
    Load the bureau balance data
    :param in_dir:  The data dir
    :return: a dataframe with the engineered bureau balance information
    """
    
    buro_bal = pd.read_csv(in_dir + '/bureau_balance.csv')
    logger.debug('Bureau bal shape : %s' % buro_bal.shape)

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

    logger.debug('Merge with buro avg')
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

    return avg_buro

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

    return avg_prev


def load_installments_data(in_dir):
    logger.debug('Reading Installments')
    inst = pd.read_csv(in_dir + '/installments_payments.csv')
    nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    avg_inst = inst.groupby('SK_ID_CURR').mean()
    avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]
    return avg_inst


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
    return avg_cc_bal


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
    return avg_pos


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

    logger.debug('Read data and test')
    data = pd.read_csv(in_dir + '/application_train.csv')
    test = pd.read_csv(in_dir + '/application_test.csv')
    logger.debug('Data Shape: %s, Test Shape: %s' %(data.shape, test.shape))

    y = data['TARGET']
    del data['TARGET']

    categorical_feats = [
        f for f in data.columns if data[f].dtype == 'object'
    ]
    logger.debug(categorical_feats)
    for f_ in categorical_feats:
        data[f_], indexer = pd.factorize(data[f_])
        test[f_] = indexer.get_indexer(test[f_])

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
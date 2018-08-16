import logging

import pandas as pd
import numpy as np
import gc

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, Imputer


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def load_bureau_data(in_dir, nan_as_category=True):
    """
    Load the bureau balance data
    :param in_dir:  The data dir
    :return: a dataframe with the engineered bureau balance information
    """
    logger = logging.getLogger(__name__)
    logger.debug('Loading bureau data...')

    bureau = pd.read_csv(in_dir + '/bureau.csv')

    bb = pd.read_csv(in_dir + '/bureau_balance.csv')
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    logger.info('Loaded bureau data')
    return bureau_agg.fillna(0)


def load_previous_applications(in_dir):
    """
    Load the previous application data and average i
    :param in_dir:
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.debug('Loading previous applications')
    prev = pd.read_csv(in_dir + '/previous_application.csv')
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    # Bug fix for bad data. Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERCENT'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERCENT': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    logger.info('Loaded previous applications')
    return prev_agg.fillna(0)


def load_installments_data(in_dir):
    logger = logging.getLogger(__name__)
    logger.debug('Loading Installments...')
    ins = pd.read_csv(in_dir + '/installments_payments.csv')
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTALL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    logger.info('Loaded installments')
    return ins_agg.fillna(0)


def load_credit_card_data(in_dir):
    logger = logging.getLogger(__name__)
    logger.debug('Loading creadit card data..')
    cc = pd.read_csv(in_dir + '/credit_card_balance.csv')
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    logger.info('Loaded creadit card data')
    return cc_agg


def load_credit_card_data_old(in_dir):
    logger = logging.getLogger(__name__)
    logger.debug('Loading CC balance..')
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
    logger.info('Loaded CC balance')
    return avg_cc_bal.fillna(0)


def load_pos_data(in_dir):
    logger = logging.getLogger(__name__)
    logger.debug('Loading POS_CASH...')
    pos = pd.read_csv(in_dir + '/POS_CASH_balance.csv')
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    logger.info("Loaded POS Cash")
    return pos_agg.fillna(0)


def load_poly_features(df_train, df_test):
    """
    Create some interaction terms based on some of the highest correlated features.
    See https://en.wikipedia.org/wiki/Interaction_(statistics)
    Subsesquent analysis of these showed they were more correlated (either +ve or -ve) with the target.
    :param df_train: The training data frame
    :param df_test: The test data frame
    :return: df_poly_features, df_poly_features_test - The training polynomial features + the test
    """
    logger = logging.getLogger(__name__)
    logger.debug('Loading polynomial features..')
    # Make a new dataframe for polynomial features
    poly_features = df_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    poly_features_test = df_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

    # imputer for handling missing values
    imputer = Imputer(strategy='median')

    # Need to impute missing values
    poly_features = imputer.fit_transform(poly_features)
    poly_features_test = imputer.transform(poly_features_test)

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree=3)
    # Train the polynomial features
    poly_transformer.fit(poly_features)

    # Transform the features
    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)
    logger.debug('Polynomial Features shape: %s' % str(poly_features.shape))

    df_poly_features = pd.DataFrame(poly_features,
                                    columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))
    df_poly_features_test = pd.DataFrame(poly_features_test,
                                         columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                                     'EXT_SOURCE_3', 'DAYS_BIRTH']))
    df_poly_features['SK_ID_CURR'] = df_train['SK_ID_CURR']
    df_poly_features_test['SK_ID_CURR'] = df_test['SK_ID_CURR']
    logger.info('Loaded polynomial features')
    return df_poly_features, df_poly_features_test


def load_additional_features(df) -> pd.DataFrame:
    """
    Calculate some additional feature:
    - CREDIT_INCOME_PERCENT: the percentage of the credit amount relative to a client's income
    - ANNUITY_INCOME_PERCENT: the percentage of the loan annuity relative to a client's income
    - CREDIT_TERM: the length of the payment in months (since the annuity is the monthly amount due
    - DAYS_EMPLOYED_PERCENT: the percentage of the days employed relative to the client's age
    - INCOME_PER_PERSON: The income split by members of the family
    """
    logger = logging.getLogger(__name__)
    df['CREDIT_INCOME_PERCENT'] =  df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    logger.info("Loaded additional columns")
    return df


def fix_missing_cols(in_train, in_test):
    missing_cols = set(in_train.columns) - set(in_test.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        in_test[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    in_test = in_test[in_train.columns]
    return in_test


def load_train_test_data(in_dir, in_cols=None):
    logger = logging.getLogger(__name__)
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
    # Bug fix for some bad data
    df_train['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df_test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    logger.debug('Data Shape: %s, Test Shape: %s' % (df_train.shape, df_test.shape))
    logger.info("Loaded training and test data")
    return df_train, df_test, y


def load_data_dummies(in_df_train, in_df_test):
    logger = logging.getLogger(__name__)
    df_train = in_df_train
    df_test = in_df_test
    categorical_feats = [
        f for f in df_train.columns if df_train[f].dtype == 'object'
    ]
    logger.debug(categorical_feats)
    for f_ in categorical_feats:
        prefix = f_
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


def append_poly_feature(df_test, df_train, in_dir):
    df_train_poly, df_test_poly = load_poly_features(df_train, df_test)
    df_train = df_train.merge(right=df_train_poly.reset_index(), how='left', on='SK_ID_CURR')
    df_test = df_test.merge(right=df_test_poly.reset_index(), how='left', on='SK_ID_CURR')
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

import unittest
import logging.config

from src.pre_process import *


class PreProcessTest(unittest.TestCase):

    def setUp(self):
        logging.config.fileConfig("logging.conf")

    def test_load_train_test_data(self):
        df_train_pre, df_test_pre, y = load_train_test_data('.\data')
        self.assertIsNotNone(df_test_pre)
        self.assertIsNotNone(df_train_pre)
        self.assertIsNotNone(y)

    def test_load_train_test_data_all(self):
        # When
        DATA_HOME_DIR = '.\data'
        df_train_pre, df_test_pre, y = load_train_test_data(DATA_HOME_DIR)
        df_train, df_test = load_data_dummies(df_train_pre, df_test_pre)
        df_train, df_test = append_poly_feature(in_dir=DATA_HOME_DIR, df_train=df_train, df_test=df_test)
        df_train, df_test = append_bureau_data(in_dir=DATA_HOME_DIR, df_train=df_train, df_test=df_test)
        df_train, df_test = append_previous_applications(in_dir=DATA_HOME_DIR, df_train=df_train, df_test=df_test)
        df_train, df_test = append_pos_data(in_dir=DATA_HOME_DIR, df_train=df_train, df_test=df_test)
        df_train, df_test = append_credit_card_data(in_dir=DATA_HOME_DIR, df_train=df_train, df_test=df_test)
        df_train, df_test = append_installments_data(in_dir=DATA_HOME_DIR, df_train=df_train, df_test=df_test)
        # Then
        self.assertIsNotNone(df_train)
        print(str(df_train.shape))
        self.assertIsNotNone(df_test)
        print(str(df_test.shape))
        # TODO: Verify columns

    def test_load_train_test_data(self):
        cols = ['SK_ID_CURR',
                'NAME_CONTRACT_TYPE',
                'CODE_GENDER',
                'FLAG_OWN_CAR',
                'FLAG_OWN_REALTY',
                'NAME_TYPE_SUITE',
                'NAME_INCOME_TYPE',
                'NAME_EDUCATION_TYPE',
                'NAME_FAMILY_STATUS',
                'NAME_HOUSING_TYPE',
                'DAYS_REGISTRATION',
                'OWN_CAR_AGE',
                # Positively correlated top 10
                'DAYS_BIRTH',
                'REGION_RATING_CLIENT_W_CITY',
                'REGION_RATING_CLIENT',
                'DAYS_LAST_PHONE_CHANGE',
                'DAYS_ID_PUBLISH',
                'REG_CITY_NOT_WORK_CITY',
                'FLAG_EMP_PHONE',
                'REG_CITY_NOT_LIVE_CITY',
                'FLAG_DOCUMENT_3',
                # Negative correlated col
                'ELEVATORS_AVG',
                'REGION_POPULATION_RELATIVE',
                'AMT_GOODS_PRICE',
                'FLOORSMAX_MODE',
                'FLOORSMAX_MEDI',
                'FLOORSMAX_AVG',
                'DAYS_EMPLOYED',
                'EXT_SOURCE_1',
                'EXT_SOURCE_3',
                'EXT_SOURCE_2'
                ]
        df_train, df_test, y = load_train_test_data('./data', in_cols=cols)
        self.assertIsNotNone(df_train)
        print(df_train.shape)

        for x in df_train.columns:
            print("Col: %s, DType: %s" % (x, df_train[x].dtype))

        numerical_feats = [
            f for f in df_train.columns if df_train[f].dtype == 'float64' or df_train[f].dtype == 'int64'
        ]

        df_train_enc, df_test_enc = load_data_dummies(df_train, df_test)
        self.assertIsNotNone(df_train_enc)
        print(df_train_enc.shape)

        df_to_test = normalize_numericals(df_train_enc, numerical_feats)
        df_to_test.sample(5)

    def test_load_credit_card_data(self):
        df = load_credit_card_data("./data")
        self.assertIsNotNone(df)

    def test_load_bureau_data(self):
        df = load_bureau_data("./data")
        self.assertIsNotNone(df)


if __name__ == '__main__':
    unittest.main()

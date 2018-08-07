import logging.config

logging.config.fileConfig("logging.conf")

from src.pre_process import *

if __name__ == '__main__':
    x_train, x_test, y = load_model('./data')
    print(x_train.columns)
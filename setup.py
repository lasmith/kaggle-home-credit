from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Regression to predict house prices for Kaggle',
    author='Laurence Smith',
    license='MIT',
    install_requires=[
        'numpy',
        'pandas==0.23.0',
        'scikit-learn==0.19.2',
        'matplotlib==2.2.2',
        'keras==2.2.2',
        'tensorflow==1.13.1',
        'seaborn==0.9.0',
        'lightgbm==2.2.3',
    ]

)

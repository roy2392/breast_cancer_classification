
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

def cancer_encode(df):
    data_encoder = preprocessing.OrdinalEncoder(
        categories=[['T1', 'T2', 'T3', 'T4'], ['N1', 'N2', 'N3'], ['IIA', 'IIB', 'IIIA', 'IIIB', 'IIIC'],
                    ['1', '2', '3', ' anaplastic; Grade IV'], ['Regional', 'Distant'],
                    ['Negative', 'Positive'], ['Negative', 'Positive'], ['Alive', 'Dead']])
    df[['T Stage ', 'N Stage', '6th Stage', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status',
        'Status']] = data_encoder.fit_transform(df[['T Stage ', 'N Stage', '6th Stage', 'Grade', 'A Stage',
                                                    'Estrogen Status', 'Progesterone Status', 'Status']].values.reshape(
        -8, 8))
    df.Grade = df.Grade + 1

    cat_var = ['Race', 'Marital Status']
    one_hot = OneHotEncoder(sparse=False)  # , drop = 'first')
    encoder_var_array = one_hot.fit_transform(df[cat_var])
    encoder_name = one_hot.get_feature_names_out(cat_var)
    encoder_vars_df = pd.DataFrame(encoder_var_array, columns=encoder_name)
    df = pd.concat([df, encoder_vars_df], axis=1)

    return df


def cancer_features_select(df):
    df['Regional_Node_pos_%'] = 100 * df['Reginol Node Positive'] / df['Regional Node Examined']
    df.drop(['Race', 'Marital Status', 'Survival Months', 'Status','differentiate'], axis=1, inplace=True)
    return df


cancer_encoder = FunctionTransformer(cancer_encode, validate=False)
cancer_features_selector = FunctionTransformer(cancer_features_select, validate=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def get_data():
    DATA_PATH = "data/bodyPerformance.csv"
    df = pd.read_csv(DATA_PATH)

    return df


def preprocess_data(data):
    columns_to_rename = {
        'body fat_%': "body_fat_percent",
        'gripForce': "grip_force",
        'sit and bend forward_cm': 'sit_and_bend_forward_cm',
        'sit-ups counts': "sit_ups_counts",
        'broad jump_cm': 'broad_jump_cm',
    }
    data.rename(columns=columns_to_rename, inplace=True)
    data.drop_duplicates(inplace=True)
    for column in data.columns:
        if "cm" in column:
            data[column[:-2] + "m"] = data[column] / 100
            data.drop(column, inplace=True, axis=1)

    data['bmi'] = data.weight_kg / np.power(data.height_m, 2)

    siralama = [
        'age',
        'gender',
        'weight_kg',
        'height_m',
        'bmi',
        'body_fat_percent',
        'grip_force',
        'sit_ups_counts',
        'sit_and_bend_forward_m',
        'broad_jump_m',
        'diastolic',
        'systolic',
        'class',
    ]
    data = data[siralama]
    data.rename(columns={"gender_M": 'gender'}, inplace=True)
    data.gender = data.gender.astype('category')
    data = pd.get_dummies(data=data, columns=['gender'], drop_first=True)
    encode = LabelEncoder()
    data['encoded_class'] = encode.fit_transform(data['class'])
    data.encoded_class = data.encoded_class.astype("category")

    data.drop(columns=['class'], inplace=True)
    return data


def preprocess():
    data = get_data()
    data = preprocess_data(data)
    return data


if __name__ == '__main__':
    preprocess()

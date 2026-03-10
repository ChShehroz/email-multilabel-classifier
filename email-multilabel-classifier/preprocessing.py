import pandas as pd
import config
from sklearn.model_selection import train_test_split


def load_data():
    df = pd.read_csv(config.DATA_PATH)
    return df

def preprocess_data(df):

    df = df.dropna()

    X = df[config.TEXT_COLUMN]

    y2 = df[config.TYPE2_COLUMN]
    y3 = df[config.TYPE3_COLUMN]
    y4 = df[config.TYPE4_COLUMN]

    return X, y2, y3, y4


def split_data(X, y2, y3, y4):

    return train_test_split(
        X,
        y2,
        y3,
        y4,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
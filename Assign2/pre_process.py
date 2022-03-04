import glob
import pandas as pd

from sklearn.model_selection import train_test_split


def split(dataset_path, is_shuffle=False, save_path=None):

    # Load datalist into DataFrame
    df = pd.read_csv (dataset_path + 'news.csv', usecols= ['title','text','label'])

    # Shuffle
    if is_shuffle:
        df = df.sample(frac=1)
    # Split the dataset
    df_train, df_test = train_test_split(df, test_size=0.2)
    # Save DataFrame to csv file.
    if save_path is not None:
        with open(save_path + 'train.csv', 'w', encoding='utf8') as f:
            df_train.to_csv(f)
        with open(save_path + 'test.csv', 'w', encoding='utf8') as f:
            df_test.to_csv(f)
    # Return the dataframe.
    return df_train, df_test


if __name__ == '__main__':
    split('G:\\machine learning\\Assign2\\', True, 'G:\\machine learning\\Assign2\\')
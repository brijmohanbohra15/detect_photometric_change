# fetch path from config.yaml file
import os

import pandas as pd
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def load_data(era_folder_path):
    """
    Load data from the given era folder path.
    Args:
        era_folder_path: The path to the era folder.
    Returns:
        df_all: The concatenated dataframe of all the objects.
    """
    files = [f for f in os.listdir(era_folder_path) if f.endswith(".csv")]
    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(era_folder_path, file))
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    # change timestamp to datetime
    df_all['timestamp'] = pd.to_datetime(
        df_all['timestamp'],
        utc=True,
        errors='coerce'
    )
    return df_all


if __name__ == "__main__":
    era_folder_path = config['data']['raw_path']
    df_all = load_data(era_folder_path)
    print(df_all.head())

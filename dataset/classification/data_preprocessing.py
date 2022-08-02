import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_data_info(base_directory):
    """
    In this funciton a dataloader is created for data loader
    """
    data = pd.DataFrame(columns=["name", "id", "image"])

    _id = 0

    with tqdm(os.listdir(base_directory), unit="directory") as t_directories:
        for directory in t_directories:
            if directory[0] != ".":
                for image in os.listdir(os.path.join(base_directory, directory)):
                    data = data.append({
                        "name": directory,
                        "id": _id,
                        "image": image,
                    }, ignore_index=True
                    )
                _id += 1

    return data


def train_val_split(train_data, num_sample_val):
    labels = train_data["name"].unique().tolist()

    train_data["mode"] = "train"

    for label in labels:
        train_data_label = train_data[train_data["name"] == label]
        train_data_label_index = train_data_label.sample(
            n=num_sample_val, random_state=np.random.RandomState()).index

        train_data.loc[train_data_label_index, "mode"] = "val"

    return train_data


def _main():
    train_data_path = "./train1000"
    test_data_path = "./test1000"

    print("Train Data Processing: ")
    train_data = get_data_info(train_data_path)
    train_data = train_val_split(train_data, num_sample_val=4)

    print("Test Data Processing: ")
    test_data = get_data_info(test_data_path)

    train_data.to_csv("./train.csv")
    test_data.to_csv("./test.csv")


if __name__ == "__main__":
    _main()

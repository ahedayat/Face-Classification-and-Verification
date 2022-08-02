import os
import random
import torch
import torchvision
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageOps


class ClassifierDataLoader(Dataset):
    """
    A dataloader for Classifier Network
    """

    def __init__(self, base_dir, df_path, transformation=None, gray=False):
        """
        Parameters :
            - base_dir: path to base directory of data
            - df_path: path to the dataframe of data
            - transformation: torchvision.transforms
        """

        super().__init__()

        self.base_dir = base_dir
        self.transformation = transformation
        self.gray = gray

        self.df = pd.read_csv(df_path)
        self.df.drop(columns="Unnamed: 0", inplace=True)

        self.df["path"] = self.df.apply(
            lambda x: os.path.join(base_dir, x[0], x[2]), axis=1)

        self.labels = self.df["name"].unique().tolist()
        self.index_2_label = {ix: label for ix,
                              label in enumerate(self.labels)}
        self.index_2_labe = {label: ix for ix, label in enumerate(self.labels)}

        self.images = list(zip(self.df["path"], self.df["id"]))
        random.shuffle(self.images)

        self.num_categories = len(self.df["id"].unique())

    def __getitem__(self, index):
        """
        In this function, an image and its one-hot label is returned.
        """

        img_path, img_cat = self.images[index]
        image = Image.open(img_path)
        if self.gray:
            image = ImageOps.grayscale(image)

        x = image

        # x = torchvision.io.read_image(img_path) / 255
        if self.transformation is not None:
            x = self.transformation(x)

        y = torch.zeros(self.num_categories)
        y[img_cat] = 1

        return x, y

    def data_class(self, label):
        """
            Get all data of a class with index=`class_index`
        """
        # label = self.index_2_label[class_id]
        images = None

        df_label = self.df[self.df["name"] == label]

        for img_path in df_label["path"].tolist():
            image = Image.open(img_path)
            if self.gray:
                image = ImageOps.grayscale(image)

            image = self.transformation(image)

            if images is None:
                images = image
            else:
                images = torch.concat([images, image])

        return images

    def get_classes(self):
        """
            Return all classes of data 
        """
        return self.labels

    def __len__(self):
        """
        `len(.)` function return number of data in dataset
        """
        return len(self.images)

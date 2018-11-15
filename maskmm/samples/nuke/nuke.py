from os.path import join, expanduser
import os
import pandas as pd
import numpy as np
from .dataset import Dataset
from .config import Config
config = Config()

DATA = join(expanduser("~"), "data", "nuke")

def get_data():
    # create validation sample
    pvalid = .2
    trainpath = join(DATA, "stage1_train")
    df = pd.DataFrame(os.listdir(trainpath), columns=["image"])
    df["subset"] = np.random.random(len(df))>pvalid
    df.loc[df.subset==True, "subset"] = "train"
    df.loc[df.subset==False, "subset"] = "valid"

    df.to_pickle(join(DATA, "subset.pkl"))
    df.subset.value_counts()

    train_ds = Dataset(config)
    train_ds.load_nuke(trainpath, "train")
    train_ds.prepare()

    val_ds = Dataset(config)
    val_ds.load_nuke(trainpath, "valid")
    val_ds.prepare()

    return train_ds, val_ds
""" run the train/valid pipeline on nuke data """

import os
import torch
import numpy as np
import pandas as pd

from maskmm.models.maskrcnn import MaskRCNN
from maskmm.learner import Learner

from maskmm.datasets.nuke.config import Config
from maskmm.datasets.nuke.dataset import Dataset

from os.path import join, expanduser
import yaml
import logging
from logging.config import dictConfig
dictConfig(yaml.load(open(join(expanduser("~"), "logging.yaml"))))
log = logging.getLogger()

# configure
ROOT_DIR = "/home/ubuntu/maskmm"
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "data/models/mask_rcnn_coco.pth")
DATA = join(expanduser("~"), "data", "nuke")
config = Config()

# create validation sample
np.random.seed(0)
pvalid = .2
trainpath = join(DATA, "stage1_train")

df = pd.DataFrame(os.listdir(trainpath), columns=["image"])
df["subset"] = np.random.random(len(df)) > pvalid
df.loc[df.subset, "subset"] = "train"
df.loc[~df.subset, "subset"] = "valid"
df.to_pickle(join(DATA, "subset.pkl"))
df.subset.value_counts()

# create datasets
dataset_train = Dataset(config)
dataset_train.load_nuke(trainpath, "train")
dataset_train.prepare()

dataset_val = Dataset(config)
dataset_val.load_nuke(trainpath, "valid")
dataset_val.prepare()

# create model with pretrained weights except for heads
model = MaskRCNN(model_dir=MODEL_DIR, config=config).to(config.DEVICE)
model.initialize_weights()
params = torch.load(COCO_MODEL_PATH)
params.pop('classifier.linear_class.weight')
params.pop("classifier.linear_bbox.weight")
params.pop("mask.conv5.weight")
params.pop('classifier.linear_class.bias')
params.pop("classifier.linear_bbox.bias")
params.pop("mask.conv5.bias")
model.load_state_dict(params, strict=False)

# train
learner = Learner(model, dataset_train, dataset_val)
learner.train(.001, 3, "heads")

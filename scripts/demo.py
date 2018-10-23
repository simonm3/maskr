import os
import glob
import random
import skimage.io
import matplotlib.pyplot as plt

from maskmm.datasets.coco.config import Config
from maskmm.models import maskrcnn
from maskmm.utils import visualize

import torch
from os.path import join

# project root
ROOT_DIR = os.path.abspath(join(__file__, os.pardir, os.pardir))
# logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "data/models/mask_rcnn_coco.pth")
# target images
IMAGE_DIR = os.path.join(ROOT_DIR, "data/images")

# Create model with coco weights
config = Config()
config.display()
model = maskrcnn.MaskRCNN(model_dir=MODEL_DIR, config=config)

model.load_state_dict(torch.load(COCO_MODEL_PATH))

# Load a random image from the images folder
f = random.choice(glob.glob(join(IMAGE_DIR, "*")))
image = skimage.io.imread(f)

# Run detection
results = model.detect([image])

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            config.CLASS_NAMES, r['scores'])
plt.show()
import os
import numpy as np
from os.path import join
from PIL import Image
from maskmm.datagen.dataset import Dataset

class Dataset(Dataset):

    def load_nuke(self, datapath, train):
        self.add_class("dsb", 1, "cell")

        path = join(datapath, train)
        for i in os.listdir(path):
            imagepath = join(path, i, "images", i+".png")
            img = Image.open(imagepath)

            self.add_image("dsb", image_id=i, path=imagepath,
                           width=img.width, height=img.height)

    def load_mask(self, image_id):
        """ return masks = [height, width, n]
            class_ids = [n]
        """
        maskpath = os.path.abspath(join(self.image_info[image_id]['path'],
                                   os.pardir, os.pardir,
                                   "masks"))
        maskfiles = [join(maskpath, m) for m in os.listdir(maskpath)]
        masks = [np.array(Image.open(m)) for m in maskfiles]
        class_ids = np.ones(len(masks), np.int32)

        if not masks:
            masks = np.empty([0, 0, 0])
            class_ids = np.empty([0], np.int32)

        masks = np.array(masks).transpose(1,2,0)
        return masks, class_ids

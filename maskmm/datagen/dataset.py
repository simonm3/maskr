import skimage
from skimage.io import imread
import numpy as np
import torch
from torch.utils.data import Dataset
from maskmm.utils import box_utils, image_utils
from maskmm.datagen.rpn_targets import build_rpn_targets
import random
from maskmm.mytools import *

import logging
log = logging.getLogger()


class Dataset(Dataset):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:
    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...
    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, config, class_map=None, augment=True):
        self.config = config
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

        self.augment = augment

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.
        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.
        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.
        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = imread(self.image_info[image_id]['path'])

        # If grayscale or rgba then convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)*255
        elif image.shape[-1] == 4:
            image = skimage.color.rgba2rgb(image)*255
        return image

    def load_mask(self, image_id):
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids

    def __getitem__(self, image_index):
        """ return image, rpn_targets and ground truth """
        image_id = self.image_ids[image_index]
        image, image_metas, gt_class_ids, gt_boxes, gt_masks = \
            self.load_image_gt(image_id, use_mini_mask=self.config.USE_MINI_MASK)

        # If no instances then skip. e.g. image has none of classes we care about.
        if gt_class_ids.eq(0).all():
            return None

        # If too many instances than subsample.
        if len(gt_boxes) > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]

        # image and masks
        image = image_utils.mold_image(image, self.config)
        image_metas = torch.tensor(image_metas)
        gt_masks = gt_masks.permute(2, 0, 1).float()

        # rpn_targets
        rpn_match, rpn_bbox = build_rpn_targets(self.config.ANCHORS, gt_class_ids, gt_boxes, self.config)

        return image, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks

    def __len__(self):
        return len(self.image_ids)

    def load_image_gt(self, image_id, use_mini_mask=False):
        """Load and return ground truth data for an image (image, mask, bounding boxes).

        use_mini_mask: If False, returns full-size masks that are the same height
            and width as the original image. These can be big, for example
            1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
            224x224 and are generated by extracting the bounding box of the
            object and resizing it to MINI_MASK_SHAPE.

        Returns:
        image: [height, width, 3]
        shape: the original shape of the image before resizing and cropping.
        class_ids: [instance_count] Integer class IDs
        bbox: [instance_count, (y1, x1, y2, x2)]
        mask: [height, width, instance_count]. The height and width are those
            of the image unless use_mini_mask is True, in which case they are
            defined in MINI_MASK_SHAPE.
        """
        # Load image and mask
        image = self.load_image(image_id)
        mask, class_ids = self.load_mask(image_id)
        class_ids = torch.tensor(class_ids, dtype=torch.float)
        shape = image.shape

        # resize image and mask
        image, window, scale, padding = image_utils.resize_image(image, self.config)
        mask = image_utils.resize_mask(mask, scale, padding)

        # augment image and mask
        if self.augment:
            image, mask = image_utils.augment(image, mask)

        # Bounding boxes. some boxes might be all zeros if the corresponding mask got cropped out.
        bbox = box_utils.extract_bboxes(mask)

        # compress masks to reduce memory usage
        if use_mini_mask:
            mask = image_utils.minimize_mask(bbox, mask, self.config.MINI_MASK_SHAPE)
        mask = torch.tensor(mask.astype(int), dtype=torch.int)

        # Active classes are those active in this dataset.
        active_class_ids = np.zeros([self.num_classes])
        source_class_ids = self.source_class_ids[self.image_info[image_id]["source"]]
        active_class_ids[source_class_ids] = 1
        image_meta = image_utils.compose_image_meta(image_id, shape, window, active_class_ids)

        return image, image_meta, class_ids, bbox, mask

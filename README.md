# Pytorch MaskRCNN

This is a Pytorch/Fastai implementation of MaskRCNN based on work by Matterport and MultiModal Learning
(see acknowledgements at bottom of page). Mainly a personal learning exercise but the simplified structure may help others wanting to understand maskrcnn.

Completed:

* Overview diagram that shows the key components of maskRCNN
* Code restructured to match the diagram to make it easy to understand and experiment with each part.
* Decluttered and cleaned up
* Works with pytorch v1 and fastai v1
* Training and prediction working on nuke data from 2017 kaggle bowl

Todo:

* More tests. How do you know it is working?
* Could the experimental test framework be useful for continuous testing? Is there a good framework already written?
* How to expand testing to keras.  i.e. how to adapt matterport for eager mode?
* Examples of outputs from each box on the diagram

## Structure

[This diagram shows how it all fits together](maskr.jpg)
Note that training and detection follow slightly different paths.

The core code reflects the diagram:
* datagen - anchors, dataset, head_targets, rpn_targets
* model - maskrcnn (whole thing), resnet (backbone), resnetFPN (feature pyramid), rpn
* filter - proposals, roialign, detections
* loss - loss functions for rpn and head

Utilities
* lib - c extensions for nms and roialign
* utils - box_utils, image_utils, visualize, batch (function decorator to process batches)
* callbacks - to tailor fastai for maskrcnn
* ipstartup - startup script for notebooks
* config - configuration constants

Samples
 * Applications typically with dataset, config, learner, notebooks

Baseline/Test (experimental)
* baseline - classes to help compare a new versus baseline version
* test - pytest functions
    - limited number of tests versus multimodal
    - tried versus matterport but original needs adapting to work with eager mode

## Installation
1. Clone this repository.

        git clone https://github.com/simonm3/maskmm.git
        
2. Build the nms and roialign binaries:
    cd maskmm/maskmm/lib
    ./make.sh
    
3. Install the python package and dependencies in edit mode

    cd maskr
    pip install -e .

4. Download the pretrained coco weights from [Google Drive](https://drive.google.com/open?id=1LXUgC2IZUYNEoXr05tdqyKFZY0pZyPDc).

## Acknowledgements and links

### Overviews

[Summary](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4) Useful blog that summarises and explains the key steps that led to maskrcnn.

[Intro to region based models](http://deeplearning.csail.mit.edu/instance_ross.pdf). Introduction written by Ross Girshick who created maskrcnn.

[Another intro blog](https://medium.com/ilenze-com/object-detection-using-deep-learning-for-advanced-users-part-1-183bbbb08b19) Object detection using deep learning for advanced users

[Intro to fasterRCNN](https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/) Down the rabbit hole of modern object detection

[ROI pooling intro](https://deepsense.ai/region-of-interest-pooling-explained/) Region of Interest Pooling explained.

### Papers

These are the main papers that led to maskrcnn:

[FastRCNN](https://arxiv.org/pdf/1504.08083.pdf)

[FasterRCNN](https://arxiv.org/pdf/1506.01497v3.pdf)

[MaskRCNN](https://arxiv.org/abs/1703.06870)

[Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)


### Packages

This package was based on the multimodal package which was based on matterport.

[Matterport Mask_RCNN](https://github.com/matterport/Mask_RCNN). A keras/tensorflow implementation of maskrcnn

[Multimodal learning](https://github.com/multimodallearning/pytorch-mask-rcnn). Pytorch version of matterport.

[Facebook detectron](https://github.com/facebookresearch/Detectron). Facebook super package that includes implementation of a wide range of image segmentation algorithms using Caffe2.

[Pytorch detectron](https://github.com/roytseng-tw/Detectron.pytorch). Conversion of detectron to pytorch.



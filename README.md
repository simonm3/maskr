# Pytorch MaskRCNN

This is a Pytorch/Fastai implementation of MaskRCNN based on work by Matterport and MultiModal Learning
(see acknowledgements at bottom of page). This was intended as a personal learning exercise but the end result has a simple structure that may help others wanting to understand maskrcnn.

Useful:

* Overview diagram that shows the key components of maskRCNN
* Code restructured to match the diagram and remove duplication. Should be easy to understand and to experiment with each part.
* Works with pytorch v1 and fastai v1
* Training and prediction working with batch size > 1

Needs work:

* Not sure I am using fastai in the best way. Fastai v1 has plenty of callbacks to modify behaviour. However I had to hack some stuff to get it working. Suggestions to clean this up are welcome!
* Limited testing. Trains with similar results to matterport on the nuke dataset (from 2017 Kaggle Bowl). However replicating exact results is really hard.
* Created a limited test framework to benchmark against the multimodal implementation but only on a few functions. Tried the same with matterport but the code is not compatible with tensorflow eager mode without some rewriting.
* Will be adding some examples of each component probably using the shapes dataset as this is really useful for exploring image segmentation

## Structure

[This diagram shows how it all fits together](maskr.jpg)
Note that training and detection follow slightly different paths.

The core code reflects the diagram:
* datagen - anchors, dataset, head_targets, rpn_targets, config
* model - maskrcnn (whole thing), resnet (backbone), resnetFPN (feature pyramid), rpn
* filter - proposals, roialign, detections
* loss - loss functions for rpn and head

Utilities
* lib - c extensions for nms and roialign
* utils - box_utils, image_utils, visualize, batch
* callbacks - to tailor fastai for maskrcnn
* ipstartup - startup script for notebooks

Samples
 * Applications typically with dataset, config, learner, notebooks
 * Each notebook has a linked py file to allow version control

Experimental
* test - experimental tests versus matterport/multimodal
* baseline - used for testing 

## Installation
1. Clone this repository.

        git clone https://github.com/simonm3/maskmm.git
        
2. Build the nms and roialign binaries:
    cd maskmm/maskmm/lib
    ./make.sh
    
3. Install dependencies

    cd maskr
    pip install -e .

3. Download the pretrained coco weights from [Google Drive](https://drive.google.com/open?id=1LXUgC2IZUYNEoXr05tdqyKFZY0pZyPDc).

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

[Matterport Mask_RCNN](https://github.com/matterport/Mask_RCNN). A keras/tensorflow implementation of maskrcnn

[Multimodal learning](https://github.com/multimodallearning/pytorch-mask-rcnn). Pytorch version of matterport.

[Facebook detectron](https://github.com/facebookresearch/Detectron). Facebook super package that includes implementation of a wide range of image segmentation algorithms using Caffe2.

[Pytorch detectron](https://github.com/roytseng-tw/Detectron.pytorch). Conversion of detectron to pytorch.



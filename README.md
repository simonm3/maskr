# pytorch-mask-rcnn

This is a Pytorch/Fastai implementation of MaskRCNN based mainly on work by Matterport and MultiModal Learning
(see acknowledgements at bottom of page). This was mostly a learning exercise for me but at the same time the result
has a simpler structure for others wanting to understand it.

The best bits:

* Overview diagram that shows the key components of maskRCNN
* Code restructured to match the diagram and remove duplication. Should be easy to understand and to experiment with 
each part.
* Updated to work with pytorch v1 and fastai v1
* Training and prediction working with batch size > 1

The bits that need some work:

* Not sure I am using fastai in the best way. Fastai v1 has plenty of callbacks to modify behaviour. However I still 
found I had to hack some stuff to get it working. Suggestions to clean this up are welcome!
* Limited testing. Trains with similar results to matterport on the nuke dataset (from 2017 Kaggle Bowl). 
But how do you know it is working?
* Have played around with a test framework to check against matterport/multimodal implementations. However this has 
low coverage. Hard to use this with matterport as code will not work in eager mode without significant changes.  
* Will be adding some examples of each component probably using the shapes dataset as this is really useful for 
exploring image segmentation

## Installation
1. Clone this repository.

        git clone https://github.com/simonm3/maskmm.git
        cd maskmm/maskmm/lib
        ./make.sh

 2. Download the pretrained coco weights from [Google Drive](https://drive.google.com/open?id=1LXUgC2IZUYNEoXr05tdqyKFZY0pZyPDc).


## Acknowledgements and links

### Overviews

[Summary](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4)
Useful blog that summarises and explains the key steps that led to maskrcnn.

[Intro to region based models](http://deeplearning.csail.mit.edu/instance_ross.pdf). Introduction written by 
Ross Girshick who created maskrcnn.

[Another intro blog](https://medium.com/ilenze-com/object-detection-using-deep-learning-for-advanced-users-part-1-183bbbb08b19)

[Intro to fasterRCNN](https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/)

[ROI pooling intro](https://deepsense.ai/region-of-interest-pooling-explained/)

### Papers

[FastRCNN](https://arxiv.org/pdf/1504.08083.pdf)

[FasterRCNN](https://arxiv.org/pdf/1506.01497v3.pdf)

[MaskRCNN](https://arxiv.org/abs/1703.06870)

[Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)


### Packages

[Matterport Mask_RCNN](https://github.com/matterport/Mask_RCNN). A keras/tensorflow implementation of maskrcnn

[Multimodal learning](https://github.com/multimodallearning/pytorch-mask-rcnn). Pytorch version of matterport.

[Facebook detectron]()https://github.com/facebookresearch/Detectron). Facebook super package that includes
 implementation of a wide range of image segmentation algorithms. Uses Caffe2.

[Pytorch detectron]()https://github.com/roytseng-tw/Detectron.pytorch). Conversion of detectron to pytorch.



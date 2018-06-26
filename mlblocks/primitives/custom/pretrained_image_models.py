from keras.applications import (xception, vgg16, vgg19, resnet50, inception_v3,
                                inception_resnet_v2, densenet, nasnet)
from .pretrained_image_base_block import PretrainedImageBase


class PretrainedXception(PretrainedImageBase):
    """Use pretrained image CNN (xception) to do "fine-tuning" classification
    or regression.

    Note: image width/height needs to be >71, and num channels must equal 3
    """

    base_model_preprocess_func = xception.preprocess_input
    base_model = xception.Xception(weights='imagenet',
                                   pooling='avg',
                                   include_top=False)


class PretrainedVGG16(PretrainedImageBase):
    base_model_preprocess_func = vgg16.preprocess_input
    base_model = vgg16.VGG16(weights='imagenet',
                             pooling='avg',
                             include_top=False)


class PretrainedVGG19(PretrainedImageBase):
    base_model_preprocess_func = vgg19.preprocess_input
    base_model = vgg19.VGG19(weights='imagenet',
                             pooling='avg',
                             include_top=False)


class PretrainedResNet50(PretrainedImageBase):
    base_model_preprocess_func = resnet50.preprocess_input
    base_model = resnet50.ResNet50(weights='imagenet',
                                   pooling='avg',
                                   include_top=False)


class PretrainedInceptionV3(PretrainedImageBase):
    base_model_preprocess_func = inception_v3.preprocess_input
    base_model = inception_v3.InceptionV3(weights='imagenet',
                                          pooling='avg',
                                          include_top=False)


class PretrainedInceptionResNetV2(PretrainedImageBase):
    base_model_preprocess_func = inception_resnet_v2.preprocess_input
    base_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet',
                                                       pooling='avg',
                                                       include_top=False)


class PretrainedDenseNet121(PretrainedImageBase):
    base_model_preprocess_func = densenet.preprocess_input
    base_model = densenet.DenseNet121(weights='imagenet',
                                      pooling='avg',
                                      include_top=False)


class PretrainedDenseNet169(PretrainedImageBase):
    base_model_preprocess_func = densenet.preprocess_input
    base_model = densenet.DenseNet169(weights='imagenet',
                                      pooling='avg',
                                      include_top=False)


class PretrainedDenseNet201(PretrainedImageBase):
    base_model_preprocess_func = densenet.preprocess_input
    base_model = densenet.DenseNet201(weights='imagenet',
                                      pooling='avg',
                                      include_top=False)


class PretrainedNasNetLarge(PretrainedImageBase):
    base_model_preprocess_func = nasnet.preprocess_input
    base_model = nasnet.NASNetLarge(weights='imagenet',
                                    pooling='avg',
                                    include_top=False)


class PretrainedNasNetMobile(PretrainedImageBase):
    base_model_preprocess_func = nasnet.preprocess_input
    base_model = nasnet.NASNetMobile(weights='imagenet',
                                     pooling='avg',
                                     include_top=False)

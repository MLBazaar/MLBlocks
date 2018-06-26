from keras.applications import (densenet, inception_resnet_v2, inception_v3, nasnet, resnet50,
                                vgg16, vgg19, xception)

from .pretrained_image_base_block import PretrainedImageBase


class PretrainedXception(PretrainedImageBase):
    """Use pretrained image CNN (xception) to do "fine-tuning" classification
    or regression.

    Note: image width/height needs to be >71, and num channels must equal 3
    """

    _base_model_preprocess_func = xception.preprocess_input
    _base_model_class = xception.Xception


class PretrainedVGG16(PretrainedImageBase):
    _base_model_preprocess_func = vgg16.preprocess_input
    _base_model_class = vgg16.VGG16


class PretrainedVGG19(PretrainedImageBase):
    _base_model_preprocess_func = vgg19.preprocess_input
    _base_model_class = vgg19.VGG19


class PretrainedResNet50(PretrainedImageBase):
    _base_model_preprocess_func = resnet50.preprocess_input
    _base_model_class = resnet50.ResNet50


class PretrainedInceptionV3(PretrainedImageBase):
    _base_model_preprocess_func = inception_v3.preprocess_input
    _base_model_class = inception_v3.InceptionV3


class PretrainedInceptionResNetV2(PretrainedImageBase):
    _base_model_preprocess_func = inception_resnet_v2.preprocess_input
    _base_model_class = inception_resnet_v2.InceptionResNetV2


class PretrainedDenseNet121(PretrainedImageBase):
    _base_model_preprocess_func = densenet.preprocess_input
    _base_model_class = densenet.DenseNet121


class PretrainedDenseNet169(PretrainedImageBase):
    _base_model_preprocess_func = densenet.preprocess_input
    _base_model_class = densenet.DenseNet169


class PretrainedDenseNet201(PretrainedImageBase):
    _base_model_preprocess_func = densenet.preprocess_input
    _base_model_class = densenet.DenseNet201


class PretrainedNasNetLarge(PretrainedImageBase):
    _base_model_preprocess_func = nasnet.preprocess_input
    _base_model_class = nasnet.NASNetLarge


class PretrainedNasNetMobile(PretrainedImageBase):
    _base_model_preprocess_func = nasnet.preprocess_input
    _base_model_class = nasnet.NASNetMobile

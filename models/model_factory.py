from models.model_architectures import *
from utils.logging_config import setup_logging
import logging

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Updated factory function to return appropriate models
def get_model(dataset_name, model_type, input_shape=None, num_classes=None):
    """
    Factory function to retrieve the appropriate model based on dataset and type.
    """
    model_type = model_type.lower()
    if model_type == "cnn":
        if dataset_name.lower() == "mnist":
            return create_mnist_net(input_shape=input_shape, num_classes=num_classes)
        elif dataset_name.lower() == "fashion_mnist":
            return create_fashion_net(input_shape=input_shape, num_classes=num_classes)
        elif dataset_name.lower() == "cifar10":
            return create_cifar10_net(input_shape=input_shape, num_classes=num_classes)
        elif dataset_name.lower() == "cifar100":
            return create_cifar100_net(input_shape=input_shape, num_classes=num_classes)
        elif dataset_name.lower() == "tiny_imagenet":
            return create_tiny_imagenet_net(input_shape=input_shape, num_classes=num_classes)
        elif dataset_name.lower() == "gtsrb":
            return create_gtsrb_net(input_shape=input_shape, num_classes=num_classes)
        elif dataset_name.lower() == "signlanguage":
            return create_signlanguage_net(input_shape=input_shape, num_classes=num_classes)

    if model_type == "pretrained_resnet" and dataset_name.lower() == "tiny_imagenet":
        return load_pretrained_resnet50_tinyimagenet(input_shape=(64, 64, 3), num_classes=num_classes)

    elif model_type == "densenet40":
        return create_densenet40(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "resnet50":
        return create_resnet50(num_classes=num_classes)
    elif model_type == "efficientnet":
        return create_efficientnet_b0(num_classes=num_classes)
    elif model_type == "pretrained_efficientnet":
        if dataset_name.lower() == "cifar100":
            model = create_cifar100_pretrained_efficientnet(input_shape=input_shape, num_classes=num_classes)
            logger.info(f"Model created: {model}")
            return model

    elif model_type == "rf":
        if dataset_name.lower() == "mnist":
            return create_RF_MNIST()
        elif dataset_name.lower() == "fashion_mnist":
            return create_RF_Fashion()
        elif dataset_name.lower() == "cifar10":
            return create_RF_CIFAR()
        elif dataset_name.lower() == "cifar100":
            return create_RF_CIFAR100()
        elif dataset_name.lower() == "signlanguage":
            return create_RF_SignLanguage()
        elif dataset_name.lower() == "gtsrb":
            return create_RF_GTSRB()

    elif model_type == "gb":
        if dataset_name.lower() == "mnist":
            return create_GB_MNIST()
        elif dataset_name.lower() == "fashion_mnist":
            return create_GB_Fashion()
        elif dataset_name.lower() == "cifar10":
            return create_GB_CIFAR()
        elif dataset_name.lower() == "cifar100":
            return create_GB_CIFAR100()
        elif dataset_name.lower() == "signlanguage":
            return create_GB_SignLanguage()
        elif dataset_name.lower() == "gtsrb":
            return create_GB_GTSRB()

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


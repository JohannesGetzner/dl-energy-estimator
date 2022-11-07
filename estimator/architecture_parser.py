import math
from functools import singledispatch
from warnings import warn
import torch
import yaml
from torch import nn
from utils.architecture_utils import traverse_architecture_and_return_module_configs
from energy_channels import (
    IdentityEnergyChannel,
    LinearEnergyChannel,
    Conv2dEnergyChannel,
    MaxPooling2dEnergyChannel,
    ActivationsEnergyChannel,
    EnergyChannel
)


@singledispatch
def _parse_torch_module(module_to_parse: nn.Module, sample_input_dims, features_config, model_version,
                        batch_size) -> EnergyChannel:
    """
    singledispatch function to parse a PyTorch Module into a custom channel class
    individual implementations for each module below
    :param module_to_parse: the module to be parsed
    :param sample_input: a sample input to the module with the correct dimensions
    :return: the custom channel class or None if the there is no matching channel implementation
    """
    warn(f"Skipping Layer: No EnergyChannel implemented for {module_to_parse}")
    return IdentityEnergyChannel()


@_parse_torch_module.register
def _(module_to_parse: nn.Linear, sample_input_dims, features_config, model_version,
      batch_size) -> LinearEnergyChannel:
    return LinearEnergyChannel(
        batch_size=batch_size,
        model_version=model_version,
        features_config=features_config,
        input_size=module_to_parse.in_features,
        output_size=module_to_parse.out_features,
    )


@_parse_torch_module.register
def _(module_to_parse: nn.Conv2d, sample_input_dims: torch.Tensor, features_config: {},
      model_version, batch_size) -> Conv2dEnergyChannel:
    return Conv2dEnergyChannel(
        batch_size=batch_size,
        model_version=model_version,
        features_config=features_config,
        in_channels=module_to_parse.in_channels,
        out_channels=module_to_parse.out_channels,
        padding=module_to_parse.padding[0],
        stride=module_to_parse.stride[0],
        kernel_size=module_to_parse.kernel_size[0],
        image_size=sample_input_dims[2]
    )


@_parse_torch_module.register
def _(module_to_parse: nn.MaxPool2d, sample_input_dims: torch.Tensor, features_config: {},
      model_version, batch_size) -> MaxPooling2dEnergyChannel:
    return MaxPooling2dEnergyChannel(
        batch_size=batch_size,
        model_version=model_version,
        features_config=features_config,
        in_channels=sample_input_dims[1],
        stride=module_to_parse.stride,
        kernel_size=module_to_parse.kernel_size,
        image_size=sample_input_dims[2],
        padding=module_to_parse.padding
    )


@_parse_torch_module.register
def _(module_to_parse: nn.ReLU, sample_input_dims: torch.Tensor, features_config: {},
      model_version, batch_size) -> ActivationsEnergyChannel:
    return ActivationsEnergyChannel(
        batch_size=batch_size,
        model_version=model_version,
        features_config=features_config,
        input_size=math.prod(sample_input_dims),
        activation_type='relu'
    )


@_parse_torch_module.register
def _(module_to_parse: nn.Sigmoid, sample_input_dims: torch.Tensor, features_config: {},
      model_version, batch_size) -> ActivationsEnergyChannel:
    return ActivationsEnergyChannel(
        batch_size=batch_size,
        model_version=model_version,
        features_config=features_config,
        input_size=math.prod(sample_input_dims),
        activation_type='sigmoid'
    )


@_parse_torch_module.register
def _(module_to_parse: nn.Tanh, sample_input_dims: torch.Tensor, features_config: {},
      model_version, batch_size) -> ActivationsEnergyChannel:
    return ActivationsEnergyChannel(
        batch_size=batch_size,
        model_version=model_version,
        features_config=features_config,
        input_size=math.prod(sample_input_dims),
        activation_type='tanh'
    )


@_parse_torch_module.register
def _(module_to_parse: nn.Softmax, sample_input_dims: torch.Tensor, features_config: {},
      model_version, batch_size) -> ActivationsEnergyChannel:
    return ActivationsEnergyChannel(
        batch_size=batch_size,
        model_version=model_version,
        features_config=features_config,
        input_size=math.prod(sample_input_dims),
        activation_type='softmax'
    )


def parse_architecture(architecture: nn.Module, batch_size: int) -> [EnergyChannel]:
    with open('../estimation_config.yaml', "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    modules = traverse_architecture_and_return_module_configs(architecture, by_type=False)
    channels = []
    for m, input_shape, module_idx in modules:
        if str(type(m).__name__).lower() in ['sigmoid', 'relu', 'softmax', 'tanh']:
            features_config = config["model_configurations"]['activations']['features_config']
            model_version = config["model_configurations"]['activations']['model_version']
        elif str(type(m).__name__).lower() in config["model_configurations"].keys():
            features_config = config["model_configurations"][str(type(m).__name__).lower()]['features_config']
            model_version = config["model_configurations"][str(type(m).__name__).lower()]['model_version']
        else:
            features_config = {}
            model_version = ''
        channel = _parse_torch_module(m, input_shape, features_config, model_version, batch_size)
        channels.append(channel)
    return channels


if __name__ == '__main__':
    from torchvision.models import resnet18

    a = resnet18(weights=None)
    parse_architecture(a, 5)

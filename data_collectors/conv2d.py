import random

import torch
from torch.nn.modules import Conv2d
from torchvision import models
from data_collectors._data_collector import DataCollector
from utils.architecture_utils import traverse_architecture_and_return_module_configs


class Conv2dDataCollector(DataCollector):

    def __init__(self,
                 module_param_configs,
                 output_path="./out.csv",
                 sampling_cutoff=500,
                 num_repeat_config=1,
                 random_sampling=True,
                 configs_from_architectures=None,
                 sampling_timeout=30,
                 ):
        super(Conv2dDataCollector, self).__init__(
            module_param_configs,
            sampling_timeout,
            sampling_cutoff,
            num_repeat_config,
            random_sampling,
            output_path
        )
        self.configs_from_architectures = configs_from_architectures

    def get_conv2d_configs_from_architectures(self) -> [dict]:
        """
        traverses an architecture like VGG11 and extracts the configuration of its Conv2D modules
        :return: a list of Conv2D configuration extracted from architectures
        """
        conv2d_configs = []
        for a in self.configs_from_architectures:
            architecture = getattr(models, a)(weights=None)
            modules = traverse_architecture_and_return_module_configs(architecture, Conv2d)
            for module, input_shape, layer_idx in modules:
                new_config = {
                    "image_size": input_shape[2],
                    "kernel_size": module.kernel_size[0],
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "stride": module.stride[0],
                    "padding": module.padding[0],
                    "batch_size": random.choice(self.module_param_configs["batch_size"]),
                    "note": f"{a}(layer_idx:{layer_idx})"
                }
                conv2d_configs.append(new_config)
        return conv2d_configs

    def validate_config(self, module, data_dim) -> bool:
        """
        validates the current configuration. For some modules e.g. Conv2D the kernel-size cannot be larger
        than the image-size
        :param module: the PyTorch Conv2D module
        :param data_dim: the dimensions of the data
        :return: True with configuration is valid, otherwise False
        """
        if isinstance(module, torch.nn.Conv2d):
            if module.kernel_size[0] > data_dim[3]:
                return False
            elif module.stride[0] > data_dim[3]:
                return False
            else:
                return True

    def initialize_module(self, config) -> torch.nn.Conv2d:
        """
        initializes the PyTorch Conv2D module
        :param config: a dict that contains the values for the module parameters
        :return: the Conv2D module
        """
        return Conv2d(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            padding=config["padding"],
            stride=config["stride"],
            kernel_size=config["kernel_size"]
        )

    def generate_data(self, config) -> torch.Tensor:
        """
        generates a random tensor of the correct size given the configuration
        :param config: the module configuration
        :return: the tensor representing the "image"
        """
        return torch.rand((config["batch_size"], config["in_channels"], config["image_size"], config["image_size"]))

    def run(self) -> None:
        """
        starts the data collection
        """
        print("Doing random configs...")
        # self.run_data_collection()
        if self.configs_from_architectures:
            print("Doing architecture configs...")
            self.run_data_collection(custom_configs=self.get_conv2d_configs_from_architectures())

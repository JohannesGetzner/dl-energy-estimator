import random
import torch
from torch.nn import MaxPool2d
from utils.architecture_utils import traverse_architecture_and_return_module_configs
from torchvision import models
from data_collectors._data_collector import DataCollector
random.seed = 111111


class MaxPooling2dDataCollector(DataCollector):

    def __init__(self,
                 module_param_configs,
                 output_path="./out.csv",
                 sampling_cutoff=500,
                 num_repeat_config=1,
                 random_sampling=True,
                 configs_from_architectures=None,
                 sampling_timeout=30,
                 ):
        super(MaxPooling2dDataCollector, self).__init__(
            module_param_configs,
            sampling_timeout,
            sampling_cutoff,
            num_repeat_config,
            random_sampling,
            output_path
        )
        self.configs_from_architectures = configs_from_architectures

    def get_maxpooling2d_configs_from_architectures(self):
        """
        traverses the architectures such as VGG11 specified in the configuration and extracts the configuration of
        all its MaxPool2d modules
        :return: a list of MaxPool2d configurations
        """
        maxpool2d_configs = []
        for a in self.configs_from_architectures:
            architecture = getattr(models, a)(weights=None)
            modules = traverse_architecture_and_return_module_configs(architecture, MaxPool2d)
            for module, input_shape, layer_idx in modules:
                new_config = {
                    "image_size": input_shape[2],
                    "in_channels": input_shape[1],
                    "kernel_size": module.kernel_size,
                    "stride": module.stride,
                    "padding": module.padding,
                    "batch_size": random.choice(self.module_param_configs["batch_size"]),
                    "note": f"{a}(layer_idx:{layer_idx})"
                }
                maxpool2d_configs.append(new_config)
        return maxpool2d_configs
        pass

    def validate_config(self, module, data_dim) -> bool:
        """
        validates the current configuration. For some modules e.g. MaxPooling2D the kernel-size cannot be larger
        than the image-size
        :param module: the PyTorch MaxPooling2d module
        :param data_dim: the dimensions of the data
        :return: True with configuration is valid, otherwise False
        """
        if module.kernel_size > data_dim[3]:
            return False
        elif module.stride > data_dim[3]:
            return False
        elif module.padding > module.kernel_size/2:
            return False
        else:
            return True

    def initialize_module(self, config) -> torch.nn.MaxPool2d:
        """
        initializes the PyTorch MaxPooling module
        :param config: a dict that contains the values for the module parameters
        :return: the Conv2D module
        """
        return MaxPool2d(
            kernel_size=config["kernel_size"],
            padding=config["padding"],
            stride=config["stride"],
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
        self.run_data_collection()
        if self.configs_from_architectures:
            print("Doing architecture configs...")
            self.run_data_collection(custom_configs=self.get_maxpooling2d_configs_from_architectures())

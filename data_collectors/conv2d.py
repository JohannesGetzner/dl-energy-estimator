import numpy as np
import torch
from torch.nn.modules import Conv2d
from torchvision import models
from data_collectors._data_collector import DataCollector
from utils.architecture_utils import traverse_architecture_and_return_module_configs
from utils.data_utils import parse_codecarbon_output


class Conv2dDataCollector(DataCollector):

    def __init__(self,
                 module_param_configs,
                 output_path,
                 sampling_cutoff=500,
                 num_repeat_config=1,
                 random_sampling=True,
                 configs_from_architectures=None,
                 sampling_timeout=30,
                 seed=None
                 ):
        super(Conv2dDataCollector, self).__init__(
            module_param_configs,
            sampling_timeout,
            sampling_cutoff,
            num_repeat_config,
            random_sampling,
            output_path,
            seed
        )
        if seed:
            np.random.seed(seed)
        self.configs_from_architectures = configs_from_architectures

    def get_conv2d_configs_from_architectures(self):
        """
        traverses the architectures such as VGG11 specified in the configuration and extracts the configuration of
        all its Conv2d modules
        :return: a list of Conv2d configuration extracted from architectures
        """
        conv2d_configs = []
        conv2d_modules = []
        for a in self.configs_from_architectures:
            architecture = getattr(models, a)(weights=None)
            modules = traverse_architecture_and_return_module_configs(architecture, by_type=True)[torch.nn.Conv2d]
            for module, input_shape, layer_idx in modules:
                new_config = ({
                    "batch_size": np.random.choice(self.module_param_configs["batch_size"]),
                    "image_size": input_shape[2],
                    "kernel_size": module.kernel_size[0],
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "stride": module.stride[0],
                    "padding": module.padding[0],
                    "note": f"{a}(layer_idx:{layer_idx})"
                })
                conv2d_configs.append(new_config)
                conv2d_modules.append(module)
        return conv2d_configs, conv2d_modules

    def validate_config(self, config) -> bool:
        """
        validates the current configuration. For some modules e.g. Conv2D the kernel-size cannot be larger
        than the image-size
        :config: the config to validate
        :return: True with configuration is valid, otherwise False
        """
        if config["kernel_size"] > config["image_size"]:
            return False
        elif config["stride"] > config["image_size"]:
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
        random_configs = self.generate_module_configurations(self.random_sampling, self.sampling_cutoff)
        a_configs, a_modules = [], []
        if self.configs_from_architectures:
            a_configs, a_modules = self.get_conv2d_configs_from_architectures()
        self.print_data_collection_info(random_configs + a_configs)

        print("Doing random configs...")
        modules = [self.initialize_module(config) for config in random_configs]
        self.run_data_collection_multiple_configs(random_configs, modules)

        if self.configs_from_architectures:
            print("Doing architecture configs...")
            self.run_data_collection_multiple_configs(a_configs, a_modules)
        parse_codecarbon_output(self.output_path)

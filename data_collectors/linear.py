import torch
import random
from torch.nn.modules import Linear
from torchvision import models
from data_collectors._data_collector import DataCollector
from utils.architecture_utils import traverse_architecture_and_return_module_configs


class LinearDataCollector(DataCollector):

    def __init__(self,
                 module_param_configs,
                 output_path="./out.csv",
                 sampling_cutoff=500,
                 num_repeat_config=1,
                 random_sampling=True,
                 configs_from_architectures=None,
                 sampling_timeout=30,
                 ):
        super(LinearDataCollector, self).__init__(
            module_param_configs,
            sampling_timeout,
            sampling_cutoff,
            num_repeat_config,
            random_sampling,
            output_path
        )
        self.configs_from_architectures = configs_from_architectures

    def validate_config(self, module, data_dim) -> bool:
        return True

    def get_linear_configs_from_architectures(self) -> [dict]:
        """
        traverses the architectures such as VGG11 specified in the configuration and extracts the configuration of
        all its Linear modules
        :return: a list of Linear configuration extracted from architectures
        """
        linear_configs = []
        for a in self.configs_from_architectures:
            architecture = getattr(models, a)(weights=None)
            modules = traverse_architecture_and_return_module_configs(architecture, Linear)
            for module, input_shape, layer_idx in modules:
                new_config = {
                    "input_size": module.out_features,
                    "output_size": module.out_features,
                    "batch_size": random.choice(self.module_param_configs["batch_size"]),
                    "note": f"{a}(layer_idx:{layer_idx})"
                }
                linear_configs.append(new_config)
        return linear_configs

    def initialize_module(self, config) -> torch.nn.Linear:
        """
        initializes the PyTorch Linear module
        :param config: a dict that contains the values for the module parameters
        :return: the Linear module
        """
        return Linear(
            in_features=config["input_size"],
            out_features=config["output_size"]
        )

    def generate_data(self, config) -> torch.Tensor:
        """
        generates a random tensor of the correct size given the configuration
        :param config: the module configuration
        :return: the tensor representing the "image"
        """
        return torch.rand(config["batch_size"], config["input_size"])

    def run(self) -> None:
        """
        starts the data collection
        :return:
        """
        print("Doing random configs...")
        self.run_data_collection()
        if self.configs_from_architectures:
            print("Doing architecture configs...")
            self.run_data_collection(custom_configs=self.get_linear_configs_from_architectures())

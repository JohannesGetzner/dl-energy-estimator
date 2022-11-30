import torch
import numpy as np
from torch.nn.modules import Linear
from torchvision import models
from data_collectors._data_collector import DataCollector
from utils.architecture_utils import traverse_architecture_and_return_module_configs
from utils.data_utils import parse_codecarbon_output


class LinearDataCollector(DataCollector):

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
        super(LinearDataCollector, self).__init__(
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

    def validate_config(self, config) -> bool:
        return True

    def get_linear_configs_from_architectures(self) -> [({}, torch.nn.Module)]:
        """
        traverses the architectures such as VGG11 specified in the configuration and extracts the configuration of
        all its Linear modules
        :return: a list of Linear configuration extracted from architectures
        """
        linear_configs = []
        linear_modules = []
        for a in self.configs_from_architectures:
            architecture = getattr(models, a)(weights=None)
            modules = traverse_architecture_and_return_module_configs(architecture, by_type=True)[torch.nn.Linear]
            for module, input_shape, layer_idx in modules:
                new_config = {
                    "batch_size": np.random.choice(self.module_param_configs["batch_size"]),
                    "input_size": module.in_features,
                    "output_size": module.out_features,
                    "freeText": f"architecture={a};layer_idx={layer_idx}"
                }
                linear_configs.append(new_config)
                linear_modules.append(module)
        return linear_configs, linear_modules

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
        :return: the tensor representing the generated data
        """
        return torch.rand(config["batch_size"], config["input_size"])

    def run(self) -> None:
        """
        starts the data collection
        """
        random_configs = self.generate_module_configurations(self.random_sampling, self.sampling_cutoff)
        a_configs, a_modules = [], []
        if self.configs_from_architectures:
            a_configs, a_modules = self.get_linear_configs_from_architectures()
        self.print_data_collection_info(random_configs + a_configs)
        print("Doing random configs...")
        modules = [self.initialize_module(config) for config in random_configs]
        self.run_data_collection_multiple_configs(random_configs, modules)

        if self.configs_from_architectures:
            print("Doing architecture configs...")
            self.run_data_collection_multiple_configs(a_configs, a_modules)
        parse_codecarbon_output(self.output_path)

import torch
from torch.nn.modules import Sigmoid
from data_collectors._data_collector import DataCollector
from utils.data_utils import parse_codecarbon_output


class SigmoidDataCollector(DataCollector):
    """
    Data Collector implementation for the Sigmoid PyTorch module
    """
    def __init__(self,
                 module_param_configs,
                 output_path,
                 sampling_cutoff=500,
                 num_repeat_config=1,
                 random_sampling=True,
                 sampling_timeout=30,
                 configs_from_architectures=None,
                 seed=None
                 ):
        super(SigmoidDataCollector, self).__init__(
            module_param_configs,
            sampling_timeout,
            sampling_cutoff,
            num_repeat_config,
            random_sampling,
            output_path,
            seed
        )

    def validate_config(self, config) -> bool:
        # there are no wrong configurations for this activation function
        return True

    def initialize_module(self, config) -> torch.nn.Module:
        """
        initializes the PyTorch Sigmoid module
        :param config: the current config
        :return: the corresponding activation PyTorch module
        """
        return Sigmoid()

    def generate_data(self, config) -> torch.Tensor:
        """
        generates a random tensor of the correct size given the configuration
        :param config: the module configuration
        :return: the tensor representing the "flattened" image
        """
        return torch.rand(config["batch_size"], config["input_size"])

    def run(self) -> None:
        """
        starts the data collection
        """

        print("Doing random configs...\n")
        random_configs = self.generate_module_configurations(self.random_sampling, self.sampling_cutoff)
        self.get_iterations_and_compute_time(random_configs)
        modules = [self.initialize_module(config) for config in random_configs]
        self.run_data_collection_multiple_configs(random_configs, modules)
        parse_codecarbon_output(self.output_path)

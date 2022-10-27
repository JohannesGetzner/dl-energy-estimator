import torch
from torch.nn.modules import ReLU, Sigmoid, Tanh, Softmax
from data_collectors._data_collector import DataCollector


class ActivationsDataCollector(DataCollector):

    def __init__(self,
                 module_param_configs,
                 output_path,
                 sampling_cutoff=500,
                 num_repeat_config=1,
                 random_sampling=True,
                 sampling_timeout=30,
                 activation_types=None,
                 seed=None
                 ):
        super(ActivationsDataCollector, self).__init__(
            module_param_configs,
            sampling_timeout,
            sampling_cutoff,
            num_repeat_config,
            random_sampling,
            output_path,
            seed
        )
        self.activation_types = activation_types

    def validate_config(self, config) -> bool:
        # there are no wrong configurations for activation functions
        return True

    def initialize_module(self, config) -> torch.nn.Module:
        """
        initializes the PyTorch Linear module
        :param config: the current activation type
        :return: the corresponding activation PyTorch module
        """
        if config == 'relu':
            return ReLU()
        elif config == 'sigmoid':
            return Sigmoid()
        elif config == 'softmax':
            return Softmax(dim=1)
        elif config == 'tanh':
            return Tanh()
        else:
            raise NotImplementedError(f"Activation of type {config} not implemented")

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
        """
        if not self.activation_types:
            raise NotImplementedError('at least one activation type has to be specified')

        random_configs = []
        modules = []
        for a in self.activation_types:
            random_configs += self.generate_module_configurations(self.random_sampling, self.sampling_cutoff)
            modules += [self.initialize_module(a) for i in range(self.sampling_cutoff)]
        print("Doing random configs...")
        self.run_data_collection_multiple_configs(random_configs, modules)

import torch
from torch.nn.modules import ReLU, Sigmoid, Tanh, Softmax
from data_collectors._data_collector import DataCollector


class ActivationsDataCollector(DataCollector):

    def __init__(self,
                 module_param_configs,
                 output_path="./out.csv",
                 sampling_cutoff=500,
                 num_repeat_config=1,
                 random_sampling=True,
                 configs_from_architectures=None,
                 sampling_timeout=30,
                 ):
        super(ActivationsDataCollector, self).__init__(
            module_param_configs,
            sampling_timeout,
            sampling_cutoff,
            num_repeat_config,
            random_sampling,
            output_path
        )
        self.count = 0

    def validate_config(self, module, data_dim) -> bool:
        return True

    def initialize_module(self, config) -> torch.nn.Module:
        """
        initializes the PyTorch Linear module
        :param config: a dict that contains the values for the module parameters
        :return: the Linear module
        """
        if self.count < self.sampling_cutoff / 4:
            module = ReLU()
        elif self.sampling_cutoff / 4 <= self.count < self.sampling_cutoff / 2:
            module = Sigmoid()
        elif self.sampling_cutoff / 2 <= self.count < self.sampling_cutoff / 4 * 3:
            module = Softmax(dim=1)
        else:
            module = Tanh()
        self.count += 1
        return module

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
        self.run_data_collection()

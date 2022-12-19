import torch
from torch.nn.modules import Linear
from torchvision import models
from data_collectors._data_collector import DataCollector
from utils.architecture_utils import traverse_architecture_and_return_module_configs
from tqdm import tqdm

from utils.data_utils import parse_codecarbon_output


class ArchitecturesDataCollector(DataCollector):

    def __init__(self,
                 module_param_configs,
                 output_path,
                 sampling_cutoff=500,
                 num_repeat_config=1,
                 random_sampling=True,
                 architectures=None,
                 include_module_wise_measurements=True,
                 sampling_timeout=30,
                 seed=None
                 ):
        super(ArchitecturesDataCollector, self).__init__(
            module_param_configs,
            sampling_timeout,
            sampling_cutoff,
            num_repeat_config,
            random_sampling,
            output_path,
            seed
        )
        self.architectures = architectures
        self.include_module_wise_measurements = include_module_wise_measurements

    def validate_config(self, config) -> bool:
        return True

    def initialize_module(self, config) -> torch.nn.Module:
        """
        initializes the PyTorch Linear module
        :param config: a dict that contains the values for the module parameters
        :return: the Linear module
        """
        return Linear(
            in_features=config["input_size"],
            out_features=config["output_size"]
        )

    def get_iterations_and_compute_time(self, **kwargs) -> (int, float):
        total_iters = len(self.architectures) * self.sampling_cutoff * self.num_repeat_config
        if self.include_module_wise_measurements:
            for a in self.architectures:
                a_module = getattr(models, a)(weights=None)
                a_modules = traverse_architecture_and_return_module_configs(a_module)
                total_iters += len(a_modules) * self.sampling_cutoff * self.num_repeat_config
        print("Total number of iterations: ", total_iters * self.num_repeat_config)
        compute_time_in_hours = round(total_iters * self.num_repeat_config * (self.sampling_timeout + 5) / 60 / 60, 2)
        print(f"Min. runtime: {compute_time_in_hours}h")
        return total_iters, compute_time_in_hours

    def generate_data(self, config) -> torch.Tensor:
        """
        generates a random tensor of the correct size given the configuration
        :param config: the module configuration
        :return: the tensor representing the "image"
        """
        return torch.rand((config["batch_size"], 3, 224, 224))

    def run(self) -> None:
        """
        starts the data collection
        """
        total_iters, _ = self.get_iterations_and_compute_time()
        iter_no = 0
        for a_name in self.architectures:
            print(f"Doing {a_name}...")
            a_module = getattr(models, a_name)(weights=None)
            a_modules = traverse_architecture_and_return_module_configs(a_module)
            configs = self.generate_module_configurations(self.random_sampling, self.sampling_cutoff)
            for idx, config in enumerate(configs):
                # print(f"current config:[{self.config_to_string(config)}]")
                data = self.generate_data(config)
                config["freeText"] = f"architecture={a_name};layer_idx={0}"
                self.run_data_collection_single_config(config, a_module, data, iter_no=iter_no,
                                                       num_iters=total_iters)
                iter_no += 1
                if self.include_module_wise_measurements:
                    # run module wise measurements
                    for module, input_shape, module_idx in a_modules:
                        # print(f"current config:[{self.config_to_string(config)}]")
                        config["freeText"] = f"architecture={a_name};layer_idx={module_idx + 1}"
                        if len(input_shape) == 1:
                            input_shape = config["batch_size"], input_shape[0]
                        else:
                            input_shape = config["batch_size"], input_shape[1], input_shape[2], input_shape[3]
                        data_m = torch.rand(input_shape)
                        self.run_data_collection_single_config(config, module, data_m, iter_no=iter_no,
                                                               num_iters=total_iters)
                        iter_no += 1
        print(f"100.0% done")
        parse_codecarbon_output(self.output_path)

import abc
import math
import time
import torch
from collections.abc import Iterable
from functools import reduce
from itertools import product
from operator import mul
from random import choice
from warnings import warn
from codecarbon import EmissionsTracker
from ptflops import get_model_complexity_info
from tqdm import tqdm


class DataCollector(abc.ABC):

    def __init__(
            self,
            module_param_configs,
            sampling_timeout,
            sampling_cutoff,
            num_repeat_config,
            random_sampling,
            output_path,
    ):
        """
        :param module_param_configs: dictionary which contains all possible values for a PyTorch module's parameters
        :param sampling_timeout: no. of seconds during which forward-passes are compute through the module
        :param sampling_cutoff: no. of module configurations to measure
        :param num_repeat_config: no. of times a configuration should be repeated
        :param random_sampling: if False all possible combinations will be measured (Attention: can explode quickly)
        :param output_path: specifies the desired path and name to the output file with the measurements
        """
        self.module_param_configs = module_param_configs
        self.sampling_timeout = sampling_timeout
        self.sampling_cutoff = sampling_cutoff
        self.num_repeat_config = num_repeat_config
        self.random_sampling = random_sampling
        self.output_path = output_path

    @abc.abstractmethod
    def generate_data(self, config) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def initialize_module(self, config) -> torch.nn.Module:
        pass

    @abc.abstractmethod
    def validate_config(self, module, data_dim):
        """
        validates the current configuration. e.g. Conv2D the kernel-size cannot be larger than the image-size
        """
        pass

    @staticmethod
    def count_macs(module, data_dims) -> int:
        """
        computes the MACs for a given PyTorch module
        :param: module: the PyTorch module for which the MACs should be counted
        :param: data_dims: the dimensions of the data, which are required to compute the MACs
        :return: the number of MACs for the given module
        """
        # max-pooling requires custom MACs counting as ptflops implementation does not account for stride and kernel
        if isinstance(module, torch.nn.MaxPool2d):
            m = module if isinstance(module, torch.nn.MaxPool2d) else module.layers[0]
            s = m.stride
            k = m.kernel_size
            p = m.padding
            flops = math.pow(k, 2) * math.pow(math.floor((data_dims[2] - k + 2 * p) / s + 1), 2) * data_dims[1]
            flops = flops * data_dims[0]
            # max-pooling does not use MACs, but only FLOPs -> must convert
            macs = flops / 2
            return int(macs)
        macs, params = get_model_complexity_info(module, tuple(data_dims[1:]), as_strings=False,
                                                 print_per_layer_stat=False)
        macs = macs * data_dims[0]
        return int(macs)

    @staticmethod
    def config_to_string(config) -> str:
        """
        simply creates a string representation of the module configuration dict
        :param config: the module configuration dict
        :return: string representation of config dict
        """
        config_str = ""
        for key, value in config.items():
            config_str += f"{key}:{value},"
        config_str = config_str[:-1]
        return config_str

    def generate_module_configurations(self, random_sampling=True, sampling_cutoff=0) -> [dict]:
        """
        generates the module configurations from the predefined ranges for each parameter
        :param random_sampling: specifies whether the module configurations should be sampled randomly
        :param sampling_cutoff: the number of module configs that should be generated
        :return: a list of configuration dicts
        """
        if not all(isinstance(item, Iterable) for item in self.module_param_configs.values()):
            raise ValueError("All configuration items must be iterables, even if they only include one element.")
        if random_sampling and sampling_cutoff is None:
            warn("No sampling cutoff count provided. All possible combinations will be measured.")
            sampling_cutoff = reduce(mul, (len(lst) for lst in self.module_param_configs.values()))

        config_list = []
        if random_sampling:
            for i in range(1, sampling_cutoff + 1):
                new_config = {param: choice(values) for param, values in self.module_param_configs.items()}
                new_config["note"] = "empty"
                config_list.append(new_config)
        else:
            # get all possible parameter combinations from the predefined ranges
            combinations = product(*self.module_param_configs.values())
            for comb in combinations:
                new_config = {self.module_param_configs.keys()[idx]: value for idx, value in enumerate(comb)}
                new_config["note"] = "empty"
                config_list.append(new_config)
        return config_list

    def run_data_collection(self, custom_configs=None) -> None:
        """
        starts the data-collection process
        :param custom_configs: allows the use of custom configurations
        """
        if custom_configs:
            configs = custom_configs
        else:
            configs = self.generate_module_configurations(self.random_sampling, self.sampling_cutoff)
        print("Total number of iterations: ", len(configs) * self.num_repeat_config)
        compute_time_in_hours = round(len(configs) * self.num_repeat_config * (self.sampling_timeout + 5) / 60 / 60, 2)
        print(f"Min. runtime: {compute_time_in_hours}h")

        pbar = tqdm(configs)
        for config in pbar:
            pbar.set_description(f"current config:[{self.config_to_string(config)}]")
            module = self.initialize_module(config)
            data = self.generate_data(config)
            # some modules such as e.g. Conv2d have parameter requirements i.e. kernel-size < image-size
            while not self.validate_config(module, data.shape):
                config = self.generate_module_configurations(self.random_sampling, 1)[0]
                module = self.initialize_module(config)
                data = self.generate_data(config)
            module.eval()
            try:
                for rep_no in range(1, self.num_repeat_config + 1):
                    self.run_forward_passes(config, module, data, rep_no, self.output_path)
            except:
                warn(f"An error occurred while processing this config [{config}]\n")

        print("----- Finished -----")

    def run_forward_passes(self, config, module, data, rep_no, output_path="./output.csv") -> None:
        """
        computes the forward-passes through the module and records the energy consumption
        :param rep_no: current number of the config repetition
        :param config: the current module config as dict
        :param output_path: the path to the file output location
        :param module: the current PyTorch module instance
        :param data: the data for the forward pass
        """
        config_str = self.config_to_string(config)
        tracker = EmissionsTracker(
            project_name=f"module:{type(module).__name__},rep_no:{rep_no},macs:{self.count_macs(module, data.shape)},{config_str}",
            save_to_file=True,
            output_file=output_path,
            log_level='warning',
            measure_power_secs=15
        )
        tracker.start()
        count = 0
        timeout_start = time.perf_counter()
        with torch.no_grad():
            # while timeout not reached compute as many forward-passes as possible
            while time.perf_counter() < timeout_start + self.sampling_timeout:
                module(data)
                count += 1
        # record number of forward-passes in codecarbon output
        tracker._project_name = f"{tracker._project_name},forward_passes:{count}"
        tracker.stop()

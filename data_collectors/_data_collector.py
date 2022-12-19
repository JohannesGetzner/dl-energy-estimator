import abc
import math
import time
import torch
import re
import numpy as np
from collections.abc import Iterable
from functools import reduce
from itertools import product
from operator import mul
from warnings import warn
from codecarbon import EmissionsTracker
from ptflops import get_model_complexity_info
from tqdm import tqdm
import time


class DataCollector(abc.ABC):

    def __init__(
            self,
            module_param_configs,
            sampling_timeout,
            sampling_cutoff,
            num_repeat_config,
            random_sampling,
            output_path,
            seed
    ):
        """
        :param module_param_configs: dictionary which contains all possible values for a PyTorch module's parameters
        :param sampling_timeout: no. of seconds during which forward-passes are compute through the module
        :param sampling_cutoff: no. of module configurations to measure
        :param num_repeat_config: no. of times a configuration should be repeated
        :param random_sampling: if False all possible combinations will be measured (Attention: can explode quickly)
        :param output_path: specifies the desired path and name to the output file with the measurements
        :param seed: specifies the seed for random sampling
        """
        self.module_param_configs = module_param_configs
        self.sampling_timeout = sampling_timeout
        self.sampling_cutoff = sampling_cutoff
        self.num_repeat_config = num_repeat_config
        self.random_sampling = random_sampling
        # TODO: add date to output file
        self.output_path = re.sub(r'(.+)(.csv)', r'\1' + '-raw' + r'\2', output_path)
        if seed:
            np.random.seed(seed)

    @abc.abstractmethod
    def generate_data(self, config) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def initialize_module(self, config) -> torch.nn.Module:
        pass

    @abc.abstractmethod
    def validate_config(self, config):
        """
        validates the current configuration. e.g. Conv2D the kernel-size cannot be larger than the image-size
        """
        pass

    @staticmethod
    def count_module_macs(module, data_dims) -> int:
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
            # TODO: evaluate formula with regards to the padding parameter
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

        configs_list = []
        if random_sampling:
            i = 0
            while i < sampling_cutoff:
                new_config = {param: np.random.choice(values) for param, values in self.module_param_configs.items()}
                new_config["freeText"] = ""
                if self.validate_config(new_config):
                    configs_list.append(new_config)
                    i += 1
        else:
            # get all possible parameter combinations from the predefined ranges
            combinations = product(*self.module_param_configs.values())
            for comb in combinations:
                new_config = {list(self.module_param_configs.keys())[idx]: value for idx, value in enumerate(comb)}
                new_config["freeText"] = ""
                if self.validate_config(new_config):
                    configs_list.append(new_config)
        return configs_list

    def print_data_collection_info(self, configs) -> None:
        """
        prints the total number of iterations and the approx min. time it will take to collect the data
        :param configs: the configs to be measured
        """
        print("Total number of iterations: ", len(configs) * self.num_repeat_config)
        compute_time_in_hours = round(len(configs) * self.num_repeat_config * (self.sampling_timeout + 5) / 60 / 60, 2)
        print(f"Min. runtime: {compute_time_in_hours}h")
        # sleep such that prints don't get messed up with tqdm progress bars
        time.sleep(0.1)

    def run_data_collection_multiple_configs(self, configs, modules) -> None:
        """
        runs the data-collection for multiple configurations of PyTorch nn.Modules
        :param configs: the configs dictionaries
        :param modules: the PyTorch module instances corresponding to the configs
        """
        pb = tqdm(list(zip(configs, modules)))
        for config, module in pb:
            pb.set_description(f"current config:[{self.config_to_string(config)}]")
            module.eval()
            for rep_no in range(1, self.num_repeat_config + 1):
                data = self.generate_data(config)
                # try:
                self.run_forward_passes(config, module, data, rep_no)
                # except:
                #    warn(f"An error occurred while processing this config [{config}]\n")

    def run_data_collection_single_config(self, config, module, data) -> None:
        """
        runs the data-collection for a single configuration of a PyTorch nn.Module
        :param config: the config dictionary
        :param module: the PyTorch module instance
        :param data: the data to be used for this single configuration
        :return:
        """
        module.eval()
        for rep_no in range(1, self.num_repeat_config + 1):
            # try:
            self.run_forward_passes(config, module, data, rep_no)
            # except:
            #    warn(f"An error occurred while processing this config [{config}]\n")

    def run_forward_passes(self, config, module, data, rep_no) -> None:
        """
        computes the forward-passes through the module and records the energy consumption
        :param rep_no: current number of the config repetition
        :param config: the current module config as dict
        :param module: the current PyTorch module instance
        :param data: the data for the forward pass
        """
        config_str = self.config_to_string(config)
        tracker = EmissionsTracker(
            project_name=f"module:{type(module).__name__},rep_no:{rep_no},macs:{self.count_module_macs(module, data.shape)},{config_str}",
            save_to_file=True,
            output_file=self.output_path,
            log_level='warning',
            measure_power_secs=3
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

import os
import yaml
from data_collectors._data_collector import DataCollector
from data_collectors import Conv2dDataCollector
from data_collectors.activations import ActivationsDataCollector
from data_collectors.architectures import ArchitecturesDataCollector
from data_collectors.linear import LinearDataCollector
from data_collectors.maxpooling2d import MaxPooling2dDataCollector

data_collectors = {
    "conv2d": Conv2dDataCollector,
    "linear": LinearDataCollector,
    "maxpooling2d": MaxPooling2dDataCollector,
    "activations": ActivationsDataCollector,
    "architectures": ArchitecturesDataCollector
}


def load_configuration(path) -> dict:
    """
    loads the configuration for the data-collection process from the .yaml file
    :param path: the path to the configuration file
    :return: a dict containing the configuration
    """
    # load data-collection configuration from yaml-file
    with open(path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def load_data_collectors(data_collection_config) -> {str: DataCollector}:
    """
    this method instantiates the DataCollectors specified in the configuration file
    :param data_collection_config: the configuration dict
    :return: a dict containing the DataCollector instances
    """
    collectors_dict = {}
    seed = data_collection_config['seed']
    sampling_timeout = data_collection_config['sampling_timeout']
    for module_name in data_collection_config["to_sample"]:
        config = data_collection_config["module_configurations"][module_name]
        print(f"Loading [{module_name}] data-collector")
        if not config['meta']['random_sampling']:
            print("Attention: grid-search enabled. Computing all possible (valid) parameter combinations.")
        if module_name == 'activations':
            collector = ActivationsDataCollector(
                sampling_timeout=sampling_timeout,
                seed=seed,
                module_param_configs={param: range(*value) if config['meta']['random_sampling'] else value for
                                      param, value in
                                      config['module_params'].items()},
                activation_types=config['meta']['activation_types'],
                output_path=os.path.dirname(__file__) + "/data/" + config['meta']['output_file_name'],
                sampling_cutoff=config['meta']['sampling_cutoff'],
                num_repeat_config=config['meta']['num_repeat_config'],
                random_sampling=config['meta']['random_sampling']
            )
        elif module_name == 'architectures':
            collector = ArchitecturesDataCollector(
                sampling_timeout=sampling_timeout,
                seed=seed,
                module_param_configs={param: range(*value) if config['meta']['random_sampling'] else value for
                                      param, value in
                                      config['module_params'].items()},
                architectures=config['meta']['architectures'],
                output_path=os.path.dirname(__file__) + "/data/" + config['meta']['output_file_name'],
                sampling_cutoff=config['meta']['sampling_cutoff'],
                num_repeat_config=config['meta']['num_repeat_config'],
                random_sampling=config['meta']['random_sampling'],
                include_module_wise_measurements=config['meta']['include_module_wise_measurements']
            )
        else:
            collector = data_collectors[module_name](
                sampling_timeout=sampling_timeout,
                seed=seed,
                module_param_configs={param: range(*value) if config['meta']['random_sampling'] else value for
                                      param, value in
                                      config['module_params'].items()},
                configs_from_architectures=config['meta']['configs_from_architectures'],
                output_path=os.path.dirname(__file__) + "/data/" + config['meta']['output_file_name'],
                sampling_cutoff=config['meta']['sampling_cutoff'],
                num_repeat_config=config['meta']['num_repeat_config'],
                random_sampling=config['meta']['random_sampling'],
            )
        collectors_dict[module_name] = collector
    return collectors_dict


def run_data_collection(collectors_to_run) -> None:
    """
    executes one DataCollector after the other
    :param collectors_to_run: the DataCollector instances
    """
    for c_name, c in collectors_to_run.items():
        print(f"\nStarting data-collection for {c_name}...")
        c.run()


if __name__ == '__main__':
    configuration = load_configuration(path=f"{os.path.dirname(__file__)}/data_collection_config.yaml")
    collectors = load_data_collectors(configuration)
    run_data_collection(collectors)

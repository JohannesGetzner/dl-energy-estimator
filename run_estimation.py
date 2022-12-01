import yaml
import torchvision.models as models
from torch import nn
from estimator.architecture_parser import parse_architecture


def run_estimation():
    with open('model_fitting_and_estimation_config.yaml', "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    for a_name, batch_size in config["estimation_to_compute"].items():
        architecture = getattr(models, a_name)(weights=None)
        total_energy, channel_wise_energies = compute_energy_estimate(
            architecture,
            batch_size,
            config["model_configurations"]
        )
        print(f"{a_name} with batch_size {batch_size}: {total_energy}")


def compute_energy_estimate(architecture: nn.Module, batch_size: int, config):
    """
    iterates over the channels and compute the energy estimate of each one
    :param architecture: the architecture to compute the estimate for
    :param batch_size: the batch_size of the forward pass
    :param config: the configuration from 'model_fitting_and_estimation_config.yaml'
    :return: the total estimate energy_consumption of the architecture and the channel wise estimates
    """
    channels = parse_architecture(architecture=architecture, batch_size=batch_size, config=config)
    total_energy = 0
    channel_wise_energies = []
    for channel in channels:
        channel_energy = channel.compute_energy_estimate()
        total_energy += channel_energy
        channel_wise_energies.append((channel, total_energy))
        print("\n")
    return total_energy, channel_wise_energies


if __name__ == '__main__':
    run_estimation()

from torch import nn
from .architecture_parser import parse_architecture


def compute_energy_estimate(architecture: nn.Module, batch_size: int, config):
    channels = parse_architecture(architecture=architecture, batch_size=batch_size, config=config)
    total_energy = 0
    channel_wise_energies = []
    for channel in channels:
        channel_energy = channel.compute_energy_estimate()
        total_energy += channel_energy
        channel_wise_energies.append((channel, total_energy))
    return total_energy, channel_wise_energies

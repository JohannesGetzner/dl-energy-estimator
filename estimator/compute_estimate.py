from torch import nn
from architecture_parser import parse_architecture


def compute_energy_estimate(architecture: nn.Module, batch_size: int):
    channels = parse_architecture(architecture=architecture, batch_size=batch_size)
    total_energy = 0
    channel_wise_energies = []
    for channel in channels:
        channel_energy = channel.compute_energy_estimate()
        total_energy += channel_energy
        channel_wise_energies.append((channel, total_energy))
    return total_energy, channel_wise_energies


if __name__ == '__main__':
    from torchvision.models import alexnet

    net = alexnet(weights=None)
    energy = compute_energy_estimate(net, 5)
    print(energy)

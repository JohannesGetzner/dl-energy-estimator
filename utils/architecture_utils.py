import torch
from torch import nn


def traverse_pytorch_module_recursively(module, children) -> None:
    """
    recursively traverses a Pytorch architecture and extracts all layers in the right order
    :param module: the nested module
    :param children: the list to add the module children to
    """
    if not list(module.children()):
        children.append(module)
    else:
        for m in module.children():
            traverse_pytorch_module_recursively(m, children)


def get_modules_from_architecture(a) -> [nn.Module]:
    """
    :param a: the architecture from which to extract the modules
    :return: a list of PyTorch modules extracted from the architecture
    """
    layers = []
    for idx, module in enumerate(list(a.children())):

        if not list(module.children()):
            layers.append(module)
        else:
            children = []
            traverse_pytorch_module_recursively(module, children)
            layers = layers + children
    return layers


def traverse_architecture_and_return_module_configs(a, by_type=False):
    """
    extracts all modules from an architecture and the input size of the data for that module
    :param by_type:
    :param a: the architecture
    :return: a list containing the modules in the right order or a dict containing the modules by type
    """
    modules_by_type = {}
    modules_by_order = []
    # initialize dict
    modules = get_modules_from_architecture(a)
    for module in modules:
        modules_by_type[type(module)] = []

    sample_input = torch.rand((1, 3, 244, 244))
    for idx, module in enumerate(modules):
        if isinstance(module, nn.Linear):
            sample_input = torch.flatten(sample_input)
        # tuple is (module, input_shape, module_layer_index)
        modules_by_type[type(module)].append((module, sample_input.shape, idx))
        modules_by_order.append((module, sample_input.shape, idx))
        if isinstance(module, nn.Conv2d):
            if sample_input.shape[1] != module.in_channels:
                print("INFO: number of input-channels for Conv2D don't match. This is likely due to skip connections")
                sample_input = torch.rand(
                    (sample_input.shape[0], module.in_channels, sample_input.shape[2], sample_input.shape[2]))
        module.eval()
        sample_input = module(sample_input)
    if by_type:
        return modules_by_type
    else:
        return modules_by_order

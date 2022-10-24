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


def traverse_architecture_and_return_module_configs(a, module_type):
    """
    extracts all module types from an architecture and the input size of the data for that module
    :param a: the architecture
    :param module_type: the module type to be extracted
    :return: returns all modules with the data shape from the specified module type
    """
    modules_by_type = {}
    # initialize dict
    modules = get_modules_from_architecture(a)
    for module in modules:
        modules_by_type[type(module)] = []

    sample_input = torch.rand((1, 3, 224, 224))
    for idx, module in enumerate(modules):
        if isinstance(module, nn.Linear):
            sample_input = torch.flatten(sample_input)
        # tuple is (module, input_shape, module_layer_index)
        modules_by_type[type(module)].append((module, sample_input.shape, idx))
        sample_input = module(sample_input)
    return modules_by_type[module_type]

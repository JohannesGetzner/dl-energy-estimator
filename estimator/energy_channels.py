import abc
import math
import os

import pandas as pd
import torch
import numpy as np
from joblib import load
from ptflops import get_model_complexity_info
from sklearn.preprocessing import FunctionTransformer
from dataclasses import dataclass


class EnergyChannel(abc.ABC):
    """
    The base class for all energy channels. A energy channel is our way of representing a PyTorch module, with all its
    parameters and additionally our methods for estimating and calculating the energy consumption
    """

    @classmethod
    def get_path(cls, version, base_dir, path_type):
        """
        constructs the path to the mode, pre- or postprocessor
        :param version: the version of the model, pre- or postprocessor
        :param base_dir: the base directory
        :param path_type: the type of to load: model, pre- or postprocessor
        :return: the path to the serialized model, pre- or postprocessor to load
        """
        files_list = [file for file in sorted(os.listdir(base_dir)) if cls.__name__.lower() in file.lower()]
        if len(files_list) == 0:
            print(f"    No {path_type} found.")
            return ''
        else:
            if version:
                final_path = \
                    [path for path in files_list if path.startswith(version)][0]
                if not final_path:
                    raise FileNotFoundError(f"No {path_type} for {cls.__name__} and version {version} has been found.")
            else:
                print(f"    No {path_type}-version was specified. Loading most recent model.")
                final_path = files_list[-1]
        print(f"    Loaded {path_type} --- {final_path} for {cls.__name__}")
        return os.path.join(base_dir, final_path)

    @classmethod
    def load_model(cls, model_version):
        """
        loads the model, which predicts the energy for the given channel
        :param model_version: the version of the model to load
        :return: the model
        """
        if not hasattr(cls, "_model_singleton"):
            print(f"Loading model for {cls.__name__}...")
            models_dir = os.path.join(os.path.dirname(__file__), "serialized_models/energy_models/")
            model_path = cls.get_path(model_version, models_dir, 'model')
            if model_path == '':
                model = None
            else:
                cls._model_singleton = load(model_path)
        return cls._model_singleton

    @classmethod
    def load_feature_preprocessors(cls, model_version):
        """
        loads the features preprocessor for the given channel
        :param model_version: the version of the preprocessor
        :return: the preprocessor
        """
        if not hasattr(cls, "_postprocessor_singleton"):
            print(f"Loading preprocessors for {cls.__name__}...")
            preprocessors_dir = os.path.join(os.path.dirname(__file__), "serialized_models/preprocessors/")
            preprocessor_path = cls.get_path(model_version, preprocessors_dir, 'preprocessor')
            if preprocessor_path == '':
                preprocessor = FunctionTransformer(lambda x: x)
            else:
                preprocessor = load(preprocessor_path)
            cls._preprocessor_singleton = preprocessor
        return cls._preprocessor_singleton

    @classmethod
    def load_target_variable_postprocessor(cls, model_version):
        """
        loads the energy value postprocessor for the given channel
        :param model_version:  the version of the postprocessor
        :return: the postprocessor
        """
        if not hasattr(cls, "_postprocessor_singleton"):
            print(f"Loading postprocessor for {cls.__name__}...")
            postprocessors_dir = os.path.join(os.path.dirname(__file__), "serialized_models/postprocessors/")
            postprocessor_path = cls.get_path(model_version, postprocessors_dir, 'postprocessor')
            if postprocessor_path == '':
                postprocessor = FunctionTransformer(lambda x: x)
            else:
                postprocessor = load(postprocessor_path)
            cls._postprocessor_singleton = postprocessor
        return cls._postprocessor_singleton

    @abc.abstractmethod
    def compute_energy_estimate(self):
        """
        computes the energy estimate for the given channel
        :return: the postprocessed energy value
        """
        pass

    @abc.abstractmethod
    def compute_macs(self):
        """
        computes the number of MACs for the given channel
        :return: the MACs
        """
        pass

    def construct_features(self, base_features, features_config):
        """
        given the channel, and the configuration returns the features for the model without preprocessing
        :return: a list of feature values
        """
        features = {}
        if features_config["enable_base_features"]:
            features.update(
                {f_name: getattr(self, f_name) for f_name in base_features}
            )
        if features_config["enable_log_features"]:
            features.update(
                {f"log_{f_name}": np.log1p(getattr(self, f_name)) for f_name in base_features}
            )
        if features_config["enable_macs_feature"]:
            features.update({"macs": self.compute_macs()})
        return pd.DataFrame(features, index=[0])


@dataclass
class IdentityEnergyChannel(EnergyChannel):
    """
    An identity channel to be used as a placeholder for unimplemented modules.
    Always returns 0 as the energy consumption estimate
    """
    model_version: str = ''

    def compute_macs(self):
        return 0

    def compute_energy_estimate(self):
        return 0


@dataclass
class LinearEnergyChannel(EnergyChannel):
    """
    the channel implementation of EnergyChannel for the Linear module
    """
    config: dict
    batch_size: int
    input_size: int
    output_size: int

    def compute_macs(self) -> int:
        macs, params = get_model_complexity_info(
            torch.nn.Linear(self.input_size, self.output_size),
            (self.input_size,),
            as_strings=False,
            print_per_layer_stat=False
        )
        macs *= self.batch_size
        return macs

    def compute_energy_estimate(self) -> float:
        preprocessor = self.load_feature_preprocessors(model_version=self.config["model_version"])
        postprocessor = self.load_target_variable_postprocessor(model_version=self.config["model_version"])
        features = self.construct_features(self.config["base_features"], self.config["features_config"])
        preprocessed_features = preprocessor.transform(features)
        model = self.load_model(model_version=self.config["model_version"])
        energy_estimate = model.predict(preprocessed_features) if model else 0
        return (
            postprocessor.inverse_transform([energy_estimate])[0]
        )


@dataclass
class Conv2dEnergyChannel(EnergyChannel):
    """
    the channel implementation of EnergyChannel for the Conv2d module
    """
    config: dict
    batch_size: int
    in_channels: int
    image_size: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int

    def compute_macs(self):
        macs, params = get_model_complexity_info(
            torch.nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding),
            (self.in_channels, self.image_size, self.image_size),
            as_strings=False,
            print_per_layer_stat=False
        )
        macs *= self.batch_size
        return macs

    def compute_energy_estimate(self):
        preprocessor = self.load_feature_preprocessors(model_version=self.config["model_version"])
        postprocessor = self.load_target_variable_postprocessor(model_version=self.config["model_version"])
        features = self.construct_features(self.config["base_features"], self.config["features_config"])
        preprocessed_features = preprocessor.transform(features)
        model = self.load_model(model_version=self.config["model_version"])
        energy_estimate = model.predict(preprocessed_features) if model else 0
        return postprocessor.inverse_transform([energy_estimate])[0]


@dataclass
class MaxPool2dEnergyChannel(EnergyChannel):
    """
    the channel implementation of EnergyChannel for the MaxPool2d module
    """
    config: dict
    batch_size: int
    in_channels: int
    image_size: int
    kernel_size: int
    stride: int
    padding: int

    def compute_macs(self):
        s = self.stride
        k = self.kernel_size
        flops = math.pow(k, 2) * math.pow(math.floor(((self.image_size - k + (2 * self.padding)) / s) + 1),
                                          2) * self.in_channels
        flops = flops * self.batch_size
        macs = flops / 2
        return macs

    def compute_energy_estimate(self):
        preprocessor = self.load_feature_preprocessors(model_version=self.config["model_version"])
        postprocessor = self.load_target_variable_postprocessor(model_version=self.config["model_version"])
        features = self.construct_features(self.config["base_features"], self.config["features_config"])
        preprocessed_features = preprocessor.transform(features)
        model = self.load_model(model_version=self.config["model_version"])
        energy_estimate = model.predict(preprocessed_features) if model else 0
        return (
            postprocessor.inverse_transform([energy_estimate])[0]
        )


@dataclass
class ReLUEnergyChannel(EnergyChannel):
    """
    the channel implementation of EnergyChannel for the ReLU module
    """
    batch_size: int
    input_size: int
    config: dict

    def compute_macs(self):
        macs, params = get_model_complexity_info(
            torch.nn.ReLU(),
            (self.input_size,),
            as_strings=False,
            print_per_layer_stat=False
        )
        macs *= self.batch_size
        return macs

    def compute_energy_estimate(self):
        preprocessor = self.load_feature_preprocessors(model_version=self.config["model_version"])
        postprocessor = self.load_target_variable_postprocessor(model_version=self.config["model_version"])
        features = self.construct_features(self.config["base_features"], self.config["features_config"])
        preprocessed_features = preprocessor.transform(features)
        model = self.load_model(model_version=self.config["model_version"])
        energy_estimate = model.predict(preprocessed_features) if model else 0
        return (
            postprocessor.inverse_transform([energy_estimate])[0]
        )


@dataclass
class TanhEnergyChannel(EnergyChannel):
    """
    the channel implementation of EnergyChannel for the Tanh module
    """
    batch_size: int
    input_size: int
    config: dict

    def compute_macs(self):
        return 0

    def compute_energy_estimate(self):
        preprocessor = self.load_feature_preprocessors(model_version=self.config["model_version"])
        postprocessor = self.load_target_variable_postprocessor(model_version=self.config["model_version"])
        features = self.construct_features(self.config["base_features"], self.config["features_config"])
        preprocessed_features = preprocessor.transform(features)
        model = self.load_model(model_version=self.config["model_version"])
        energy_estimate = model.predict(preprocessed_features) if model else 0
        return (
            postprocessor.inverse_transform([energy_estimate])[0]
        )


@dataclass
class SigmoidEnergyChannel(EnergyChannel):
    """
    the channel implementation of EnergyChannel for the Sigmoid module
    """
    batch_size: int
    input_size: int
    config: {}

    def compute_macs(self):
        return 0

    def compute_energy_estimate(self):
        preprocessor = self.load_feature_preprocessors(model_version=self.config["model_version"])
        postprocessor = self.load_target_variable_postprocessor(model_version=self.config["model_version"])
        features = self.construct_features(self.config["base_features"], self.config["features_config"])
        preprocessed_features = preprocessor.transform(features)
        model = self.load_model(model_version=self.config["model_version"])
        energy_estimate = model.predict(preprocessed_features) if model else 0
        return (
            postprocessor.inverse_transform([energy_estimate])[0]
        )


@dataclass
class SoftMaxEnergyChannel(EnergyChannel):
    """
    the channel implementation of EnergyChannel for the Softmax module
    """
    batch_size: int
    input_size: int
    config: dict

    def compute_macs(self):
        return 0

    def compute_energy_estimate(self):
        preprocessor = self.load_feature_preprocessors(model_version=self.config["model_version"])
        postprocessor = self.load_target_variable_postprocessor(model_version=self.config["model_version"])
        features = self.construct_features(self.config["base_features"], self.config["features_config"])
        preprocessed_features = preprocessor.transform(features)
        model = self.load_model(model_version=self.config["model_version"])
        energy_estimate = model.predict(preprocessed_features) if model else 0
        return (
            postprocessor.inverse_transform([energy_estimate])[0]
        )

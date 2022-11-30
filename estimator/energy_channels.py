import abc
import math
import os
import torch
import numpy as np
from joblib import load
from ptflops import get_model_complexity_info
from sklearn.preprocessing import FunctionTransformer
from dataclasses import dataclass


class EnergyChannel(abc.ABC):
    # TODO: only load model once per session

    @classmethod
    def get_path(cls, version, base_dir, path_type):
        """
        constructs the path to the mode, pre- or postprocessor
        :param version: the version of the model, pre- or postprocessor
        :param base_dir: the base directory
        :param path_type: the type of to load: model, pre- or postprocessor
        :return: the path to the serialized model, pre- or postprocessor to load
        """
        files_list = [
            file
            for file in sorted(os.listdir(base_dir))
        ]
        if len(files_list) == 0:
            return ''
        else:
            if version:
                final_path = [path for path in files_list if path.startswith(version)][0]
                if not final_path:
                    raise FileNotFoundError(f"No {path_type} for {cls.__name__} and version {version} has been found.")
            else:
                print(f"    No {path_type}-version was specified. Loading most recent model")
                final_path = files_list[-1]
        print(f"    Loaded {path_type} --- {final_path} for {cls.__name__}")
        return final_path

    @classmethod
    def load_model(cls, model_version):
        """
        loads the model, which predicts the energy for the given channel
        :param model_version: the version of the model to load
        :return: the model
        """
        print(f"Loading model for {cls.__name__}...")
        models_dir = os.path.join(os.path.dirname(__file__), "energy_models/")
        model_path = cls.get_path(model_version, models_dir, 'model')
        if model_path == '':
            model = None
        else:
            model = load(model_path)
        return model

    @classmethod
    def load_feature_preprocessors(cls, model_version):
        """
        loads the features preprocessor for the given channel
        :param model_version: the version of the preprocessor
        :return: the preprocessor
        """
        print(f"Loading preprocessors for {cls.__name__}...")
        preprocessors_dir = os.path.join(os.path.dirname(__file__), "feature_preprocessors/")
        preprocessor_path = cls.get_path(model_version, preprocessors_dir, 'preprocessor')
        if preprocessor_path == '':
            preprocessor = FunctionTransformer(lambda x: x)
        else:
            preprocessor = load(preprocessor_path)
        return preprocessor

    @classmethod
    def load_target_variable_postprocessor(cls, model_version):
        """
        loads the energy value postprocessor for the given channel
        :param model_version:  the version of the postprocessor
        :return: the postprocessor
        """
        print(f"Loading postprocessor for {cls.__name__}...")
        postprocessors_dir = os.path.join(os.path.dirname(__file__), "target_postprocessors/")
        postprocessor_path = cls.get_path(model_version, postprocessors_dir, 'postprocessor')
        if postprocessor_path == '':
            postprocessor = FunctionTransformer(lambda x: x)
        else:
            postprocessor = load(postprocessor_path)
        return postprocessor

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

    def construct_features(self):
        """
        given the channel, and the configuration returns the features for the model without preprocessing
        :return: a list of feature values
        """
        # TODO: consider making features a dictionary
        features = []
        # TODO: resolve property warnings
        if len(self.features_config["base_features"]) != 0:
            features += [getattr(self, feature_name) for feature_name in self.features_config["base_features"]]
            if self.features_config["enable_log_features"]:
                features += [np.log1p(feature) for feature in features]
        if self.features_config["enable_macs_feature"]:
            features += [self.compute_macs()]
        return features


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
    model_version: str
    features_config: dict
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
        preprocessor = self.load_feature_preprocessors(model_version=self.model_version)
        postprocessor = self.load_target_variable_postprocessor(model_version=self.model_version)
        features = self.construct_features()
        preprocessed_features = preprocessor.transform(
            [features]
        )
        model = self.load_model(model_version=self.model_version)
        energy_estimate = model.predict([*preprocessed_features]) if model else 0
        return (
            postprocessor.inverse_transform([energy_estimate])[0]
        )


@dataclass
class Conv2dEnergyChannel(EnergyChannel):
    """
    the channel implementation of EnergyChannel for the Conv2d module
    """
    model_version: str
    features_config: {}
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
        preprocessor = self.load_feature_preprocessors(model_version=self.model_version)
        postprocessor = self.load_target_variable_postprocessor(model_version=self.model_version)
        features = self.construct_features()
        preprocessed_features = preprocessor.transform(
            [features]
        )
        model = self.load_model(model_version=self.model_version)
        energy_estimate = model.predict([*preprocessed_features]) if model else 0
        return postprocessor.inverse_transform([energy_estimate])[0]


@dataclass
class MaxPooling2dEnergyChannel(EnergyChannel):
    """
    the channel implementation of EnergyChannel for the MaxPooling2d module
    """
    model_version: str
    features_config: {}
    batch_size: int
    in_channels: int
    image_size: int
    kernel_size: int
    stride: int
    padding: int

    def compute_macs(self):
        s = self.stride
        k = self.kernel_size
        # TODO: evaluate formula with padding
        flops = math.pow(k, 2) * math.pow(math.floor((self.image_size - k + 2 * self.padding) / s + 1),
                                          2) * self.in_channels
        flops = flops * self.batch_size
        macs = flops / 2
        return macs

    def compute_energy_estimate(self):
        preprocessor = self.load_feature_preprocessors(model_version=self.model_version)
        postprocessor = self.load_target_variable_postprocessor(model_version=self.model_version)
        features = self.construct_features()
        preprocessed_features = preprocessor.transform(
            [features]
        )
        model = self.load_model(model_version=self.model_version)
        energy_estimate = model.predict([*preprocessed_features]) if model else 0
        return (
            postprocessor.inverse_transform([energy_estimate])[0]
        )


@dataclass
class ReLUEnergyChannel(EnergyChannel):
    batch_size: int
    input_size: int
    model_version: str
    features_config: {}

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
        preprocessor = self.load_feature_preprocessors(model_version=self.model_version)
        postprocessor = self.load_target_variable_postprocessor(model_version=self.model_version)
        features = self.construct_features()
        preprocessed_features = preprocessor.transform(
            [features]
        )
        model = self.load_model(model_version=self.model_version)
        energy_estimate = model.predict([*preprocessed_features]) if model else 0
        return (
            postprocessor.inverse_transform([energy_estimate])[0]
        )


@dataclass
class TanhEnergyChannel(EnergyChannel):
    batch_size: int
    input_size: int
    model_version: str
    features_config: {}

    def compute_macs(self):
        return 0

    def compute_energy_estimate(self):
        preprocessor = self.load_feature_preprocessors(model_version=self.model_version)
        postprocessor = self.load_target_variable_postprocessor(model_version=self.model_version)
        features = self.construct_features()
        preprocessed_features = preprocessor.transform(
            [features]
        )
        model = self.load_model(model_version=self.model_version)
        energy_estimate = model.predict([*preprocessed_features]) if model else 0
        return (
            postprocessor.inverse_transform([energy_estimate])[0]
        )


@dataclass
class SigmoidEnergyChannel(EnergyChannel):
    batch_size: int
    input_size: int
    model_version: str
    features_config: {}

    def compute_macs(self):
        return 0

    def compute_energy_estimate(self):
        preprocessor = self.load_feature_preprocessors(model_version=self.model_version)
        postprocessor = self.load_target_variable_postprocessor(model_version=self.model_version)
        features = self.construct_features()
        preprocessed_features = preprocessor.transform(
            [features]
        )
        model = self.load_model(model_version=self.model_version)
        energy_estimate = model.predict([*preprocessed_features]) if model else 0
        return (
            postprocessor.inverse_transform([energy_estimate])[0]
        )


@dataclass
class SoftMaxEnergyChannel(EnergyChannel):
    batch_size: int
    input_size: int
    model_version: str
    features_config: {}

    def compute_macs(self):
        return 0

    def compute_energy_estimate(self):
        preprocessor = self.load_feature_preprocessors(model_version=self.model_version)
        postprocessor = self.load_target_variable_postprocessor(model_version=self.model_version)
        features = self.construct_features()
        preprocessed_features = preprocessor.transform(
            [features]
        )
        model = self.load_model(model_version=self.model_version)
        energy_estimate = model.predict([*preprocessed_features]) if model else 0
        return (
            postprocessor.inverse_transform([energy_estimate])[0]
        )

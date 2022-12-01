import abc
import os
import pandas as pd
from datetime import datetime
from joblib import dump

from experiments.experiments_utils import compute_log_transformed_features
from utils.data_utils import preprocess_and_normalize_energy_data


class EnergyModel():

    def __init__(
            self,
            save_to_path_models,
            save_to_path_transforms,
            config,
            data_path
    ):
        self.save_to_path_models = save_to_path_models
        self.save_to_path_transforms = save_to_path_transforms
        self.config = config
        self.data_path = data_path
        self.SEED = 1234
        self.model = None
        self.transformers_dict = None

    @abc.abstractmethod
    def fit_model(self):
        pass

    def construct_features(self, data):
        features = self.config["base_features"]
        if self.config["enable_log_features"]:
            data_with_log, param_cols_with_log = compute_log_transformed_features(
                data,
                self.config["base_features"]
            )
            data = data_with_log
            features = param_cols_with_log
        if self.config["enable_macs_feature"]:
            features += ['macs']
        return features, data

    def load_data(self, param_cols, path):
        data_unnormalized = pd.read_csv(path)
        data = preprocess_and_normalize_energy_data(data_unnormalized, param_cols, aggregate=True)
        return data

    def save_model_and_transformers(self, filename):
        """
        This function takes in the respective model and data transformers for an implemented
        class in the analyzer_classes file, and pickles them in the correct folders.
        """
        dump(
            self.model,
            os.path.join(
                self.save_to_path_models, f"{str(datetime.now().date())}_{filename}_{str(self.model)[:-2]}.joblib"
            )
        )

        if "x_preprocessors" in self.transformers_dict:
            for idx, transformer in enumerate(self.transformers_dict["x_preprocessors"]):
                dump(
                    transformer,
                    os.path.join(
                        self.save_to_path_transforms + "/preprocessors",
                        f"{str(datetime.now().date())}_{filename}_{str(transformer)[:-2]}.joblib",
                    ),
                )

        if "y_preprocessor" in self.transformers_dict:
            preprocessor = self.transformers_dict["y_preprocessor"]
            dump(
                preprocessor,
                os.path.join(
                    self.save_to_path_transforms + "/postprocessors",
                    f"{str(datetime.now().date())}_{filename}_{str(preprocessor)[:-2]}.joblib",
                ),
            )

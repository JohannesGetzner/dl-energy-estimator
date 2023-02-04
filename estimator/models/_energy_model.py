import abc
import os
from warnings import warn

import pandas as pd
from datetime import datetime
from joblib import dump
from sklearn.pipeline import Pipeline

from utils.experiments_utils import compute_log_transformed_features


class EnergyModel:
    """
    Base class for each layer-type energy predictor
    """

    def __init__(
            self,
            save_to_path_models,
            save_to_path_transforms,
            config,
            data_dir
    ):
        self.save_to_path_models = save_to_path_models
        self.save_to_path_transforms = save_to_path_transforms
        self.config = config
        self.data_dir = data_dir
        self.SEED = 1234
        self.model = None
        self.transformers_dict = None

    @abc.abstractmethod
    def fit_model(self) -> None:
        """
        the individual predictor configuration and training process method
        """
        pass

    def construct_features(self, data) -> ([], pd.DataFrame):
        """
        depending on the configuration in the .yaml file, this method constructs the feature set for each predictor to
        be trained on
        :param data: the initial raw data
        :return: returns a list of all features and a dataset where the new features have been added as columns
        """
        f_config = self.config["features_config"]
        features = []
        if f_config["enable_base_features"]:
            features += self.config["base_features"]
        if f_config["enable_log_features"]:
            data_with_log, param_cols_with_log = compute_log_transformed_features(
                data,
                self.config["base_features"]
            )
            data = data_with_log
            features = param_cols_with_log
        if f_config["enable_macs_feature"]:
            features += ['macs']
        if len(features) == 0:
            warn("No feature set specified!")
        print("Used features: ", features)
        return features, data

    def load_data(self, path) -> pd.DataFrame:
        """
        loads simply loads the data from the raw .csv files
        :param path: the path to the csv file
        :return: a pandas DataFrame containing the data
        """
        data = pd.read_csv(path)
        return data

    def save_model_and_transformers(self, filename):
        """
        This function takes in the respective model and data transformers for an implemented
        class in the analyzer_classes file, and pickles them in the correct folders.
        :param filename: the name for the corresponding model, which will be added to the file-name
        """
        dump(
            self.model,
            os.path.join(
                self.save_to_path_models, f"{str(datetime.now().date())}_{filename}_{str(self.model)[:-2]}.joblib"
            )
        )

        if self.transformers_dict["x_preprocessors"]:
            if len(self.transformers_dict["x_preprocessors"]) > 1:
                pipe = Pipeline([(f"step-{idx}", t) for idx, t in enumerate(self.transformers_dict["x_preprocessors"])])
                file_name_ext = ""
                for t in self.transformers_dict["x_preprocessors"]:
                    file_name_ext += str(t)
                dump(
                    pipe,
                    os.path.join(
                        self.save_to_path_transforms + "/preprocessors",
                        f"{str(datetime.now().date())}_{filename}_{file_name_ext}.joblib",
                    ),
                )
            else:
                transformer = self.transformers_dict["x_preprocessors"][0]
                dump(
                    transformer,
                    os.path.join(
                        self.save_to_path_transforms + "/preprocessors",
                        f"{str(datetime.now().date())}_{filename}_{str(transformer)[:-2]}.joblib",
                    ),
                )

        if self.transformers_dict["y_preprocessor"]:
            preprocessor = self.transformers_dict["y_preprocessor"]
            dump(
                preprocessor,
                os.path.join(
                    self.save_to_path_transforms + "/postprocessors",
                    f"{str(datetime.now().date())}_{filename}_{str(preprocessor)[:-2]}.joblib",
                ),
            )

import abc
import os
import pandas as pd
from datetime import datetime
from joblib import dump

from utils.data_utils import preprocess_and_normalize_energy_data


class EnergyModel():

    def __init__(
            self,
            save_to_path_models,
            save_to_path_transforms,
    ):
        self.save_to_path_models = save_to_path_models
        self.save_to_path_transforms = save_to_path_transforms
        self.SEED = 1234
        self.model = None

    @abc.abstractmethod
    def fit_model(self):
        pass

    def load_data(self, param_cols, path):
        data_unnormalized = pd.read_csv(path)
        data = preprocess_and_normalize_energy_data(data_unnormalized, param_cols, aggregate=True)
        return data

    def save_model_and_transformers(self,
                                    model,
                                    transformers_dict
                                    ):
        """
        This function takes in the respective model and data transformers for an implemented
        class in the analyzer_classes file, and pickles them in the correct folders.
        """

        dump(
            model,
            os.path.join(
                self.save_to_path_models, f"{str(datetime.now().date())}_{__file__}.joblib"
            ),
        )

        if "x" in transformers_dict:
            for idx, transformer in enumerate(transformers_dict["x"]):
                dump(
                    transformer,
                    os.path.join(
                        self.save_to_path_transforms,
                        f"{str(datetime.now().date())}_{__file__}_{idx}.joblib",
                    ),
                )

        if "y" in transformers_dict:
            dump(
                transformers_dict["y"],
                os.path.join(
                    self.save_to_path_transforms,
                    f"{str(datetime.now().date())}_{__file__}.joblib",
                ),
            )

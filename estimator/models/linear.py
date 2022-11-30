from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from estimator.models._energy_model import EnergyModel
from experiments.experiments_utils import split_data_set, apply_data_transforms, \
    fit_model, test_model


class LinearEnergyModel(EnergyModel):

    def __init__(self,
                 save_to_path_models,
                 save_to_path_transforms
                 ):
        super(LinearEnergyModel, self).__init__(
            save_to_path_models,
            save_to_path_transforms,
        )
        self.param_cols = []

    def fit_model(self):
        data = self.load_data(self.param_cols, "../../data/")
        dfs = split_data_set(data, ['macs'], self.SEED)
        transformers_dict = {
            "x_preprocessors": None,
            "y_preprocessor": MinMaxScaler()
        }
        dfs, _ = apply_data_transforms(dfs, transformers_dict)
        model, val_score, val_mse = fit_model(LinearRegression(), dfs["x_train"], dfs["y_train"], dfs["x_val"],
                                              dfs["y_val"],
                                              plot_results=False)
        y_hat, test_score, test_mse = test_model(model, dfs["x_test"], dfs["y_test"], plot_results=True)
        self.model = model
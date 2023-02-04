from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from estimator.models._energy_model import EnergyModel
from utils.experiments_utils import split_data_set, apply_data_transforms, \
    fit_model, test_model


class Conv2dEnergyModel(EnergyModel):
    """
    The class implementing the energy predictor for the Conv2d PyTorch module
    """
    def __init__(self,
                 save_to_path_models,
                 save_to_path_transforms,
                 config,
                 data_dir
                 ):
        super(Conv2dEnergyModel, self).__init__(
            save_to_path_models,
            save_to_path_transforms,
            config,
            data_dir
        )

    def fit_model(self):
        data = self.load_data(self.data_dir + self.config["data_file"])
        features, data = self.construct_features(data)
        dfs = split_data_set(data, features, self.SEED)
        transformers_dict = {
            "x_preprocessors": [],
            "y_preprocessor": MinMaxScaler()
        }
        dfs, transformers_dict = apply_data_transforms(dfs, transformers_dict)
        model, val_score, val_mse = fit_model(LinearRegression(), dfs["x_train"], dfs["y_train"], dfs["x_val"],
                                              dfs["y_val"],
                                              plot_results=False)
        y_hat, test_score, test_mse = test_model(model, dfs["x_test"], dfs["y_test"], plot_results=False)
        self.model = model
        self.transformers_dict = transformers_dict

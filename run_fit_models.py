import yaml
import os
from estimator.models import *

model_classes = {
    "conv2d": Conv2dEnergyModel,
    "linear": LinearEnergyModel,
    "maxpool2d": MaxPooling2dEnergyModel,
    "relu": ReLUEnergyModel,
}


def fit_models():
    with open('model_fitting_and_estimation_config.yaml', "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    for model_name, model_config in config['model_configurations'].items():
        model = model_classes[model_name](
            save_to_path_models="estimator/serialized_models/energy_models",
            save_to_path_transforms="./estimator/serialized_models/",
            config=model_config["features_config"],
            data_path=f"{os.getcwd()}{config['data_directory']}/{model_config['data_file_name']}"
        )
        model.fit_model()
        model.save_model_and_transformers(model_name)
        break


if __name__ == '__main__':
    fit_models()

import yaml
import os
from estimator.models import *

model_classes = {
    "conv2d": Conv2dEnergyModel,
    "linear": LinearEnergyModel,
    "maxpool2d": MaxPool2dEnergyModel,
    "relu": ReLUEnergyModel,
    "sigmoid": SigmoidEnergyModel,
    "softmax": SoftMaxEnergyModel,
    "tanh": TanhEnergyModel,
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
            config=model_config,
            data_dir=f"{os.getcwd()}{config['data_directory']}"
        )
        print(f"\n--------------------\nFitting [{model_name}] model")
        model.fit_model()
        model.save_model_and_transformers(model_name+"EnergyChannel")
        print(f"--------------------")


if __name__ == '__main__':
    fit_models()

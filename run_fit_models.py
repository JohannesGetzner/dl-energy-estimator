import yaml
from estimator.models import *

model_classes = {
    "conv2d": Conv2dEnergyModel,
    "linear": LinearEnergyModel,
    "maxpool2d": MaxPooling2dEnergyModel,
    "relu": ReLUEnergyModel,
}


def fit_models():
    with open('./estimation_config.yaml', "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    for model_name, model_config in config['model_configurations'].items():
        model = model_classes[model_name](
            save_to_path_models="estimator/serialized_models/energy_models",
            save_to_path_transforms="./estimator/serialized_models/",
            config=model_config["features_config"]
        )
        model.fit_model()
        model.save_model_and_transformers(model_name)
        break


if __name__ == '__main__':
    fit_models()

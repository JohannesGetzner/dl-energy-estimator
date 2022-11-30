import yaml
import torchvision.models as models
from estimator.compute_estimate import compute_energy_estimate

if __name__ == '__main__':
    with open('./estimation_config.yaml', "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    for a_name, batch_size in config["estimation_to_compute"].items():
        architecture = getattr(models, a_name)(weights=None)
        total_energy, channel_wise_energies = compute_energy_estimate(
            architecture,
            batch_size,
            config["model_configurations"]
        )

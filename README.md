# Accuracy is not the only Metric that matters: Estimating the Energy Consumption of Deep Learning Models

## About this project

This repository contains the sourcecode to the paper: <br>

[Accuracy is not the only Metric that matters: Estimating the Energy Consumption of Deep Learning Models](https://www.cs.cit.tum.de/daml/energy-consumption-dl/#c35468) <br>
Johannes Getzner, Bertrand Charpentier, Stephan Günnemann <br>
Spotlight, Tackling Climate Change with Machine Learning Workshop, ICLR 2023

[[Paper](https://arxiv.org/abs/2304.00897)|[Video](https://www.youtube.com/@tum-daml1874)]

## About the structure of this repository

- **./data** contains the raw file outputs by codecarbon, the '...slurm-log.out' files correspond to the log files from
  the cluster
  the collection process was run on and are used to remove potentially bad measurements.
- **./data_collectors** contains the parent *DataCollector* class and the corresponding child-class implementations for
  each module used in the data-collection process. To add a new layer e.g. AdaptiveAvgPooling, a new data-collector
  needs to be implemented.
- **./estimator** contains everything related to the final predictors on both the layer- and architecture-level
    - **./data** contains the parsed and normalized data files used for training (see the *generate_datasets.ipynb*
      notebook for details)
    - **./models** contains the custom model classes that implement the best models for each layer type
    - **./serialized_models** contains the serialized trained energy predictors, pre- and postprocessors for each
      layer-type
- **./experiments** contains a set of jupyter notebooks with the results presented in the paper
- **./utils** a set of python scripts that contain useful custom functions
- **./** besides the above, the .yaml files are required for the configuration of the data-collection, training and
  inference process. See the 'run_....py' files for more details

## Requirements

To install the necessary packages run the following command:

    pip install -r requirements.txt

codecarbon reads the Intel RAPL files for the energy information. Before the data-collection process can run, make sure
your system allows these files to be read: `sudo chmod 555 -R /sys/class/powercap/intel-rapl/` (
see [./give_RAPL_read_access.sh](https://github.com/JohannesGetzner/dl-energy-estimator/blob/main/give_RAPL_read_access.sh))

## Running the project

We developed a pipeline to (1) collect data on the energy consumption of various DL layer types and architectures such
as VGG16, (2) fit simple energy predictor models on this energy data, and (3) evaluate the energy predictors on new
data, by predicting the energy consumption of complete architectures.

### (1) Data Collection

Before attempting to start the process to collect new data, please review the *data_collection_config.yaml* file in the
root of the
repository. Subsequently, the data-collection process can be started by simply running `python run_data_collection.py`.
Note that pre-collected data is already available in the *./data* folder.

### (2) Model Fitting

Before running the model fitting procedure make sure that the parsed and normalized datasets are available in the
*./estimator/data* folder. One may also use the *./estimator/generate_datasets.ipynb* notebook to automate this
process. <br>
The model fitting procedure uses the models implemented in *./estimator/models*. If model configurations are to be
changed please review those files. <br>
Finally, before running the model-fitting process via `python run_fit_models.py`, please review the
*./model_fitting_and_estimation_config.yaml* configuration file. After the run has finished, the serialized models will
be saved automatically in
*./estimator/serialized_models*.

### (3) Inference

Given the serialized models, the system is capable of estimating the energy consumption of a variety of architectures.
By running `python run_estimation.py` the system will attempt to predict the energy consumption of the models specified
at the beginning of the *model_fitting_and_estimation_config.yaml*.

## Acknowledgments

First and foremost, I would like to express my deep gratitude to my
supervisor [Bertrand Charpentier](https://www.cs.cit.tum.de/daml/team/bertrand-charpentier/), from
the [DAML Group](https://www.cs.cit.tum.de/daml/startseite/) at the Technical University of Munich, led
by [Prof. Dr. Stephan Günnemann](https://www.cs.cit.tum.de/en/daml/team/damlguennemann/), for his unwavering support,
guidance, and valuable feedback throughout the entire duration of this project.
His insights and expertise were crucial in shaping the direction of our research and bringing it to success. Without
their constant support and encouragement, this work would not have been possible.

Additionally, I would like to acknowledge the contribution of my former group
member [Ahmed Darwish](https://github.com/Shiro-Raven), who provided valuable code during the early stages of this
project. Thank you for your help.

## Cite

    @misc{dlenergyestimator,
          title={Accuracy is not the only Metric that matters: Estimating the Energy Consumption of Deep Learning Models}, 
          author={Johannes Getzner and Bertrand Charpentier and Stephan Günnemann},
          year={2023},
          primaryClass={cs.LG},
          howpublished = {\url{https://arxiv.org/abs/2304.00897}},
          note={ICLR 2023 Workshop: Tackling Climate Change with Machine Learning by Climate Change AI}
    }

## Poster

![workshop poster](./iclr_2023_workshop_poster.jpg)

from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

from recbole.model.general_recommender import BPR, LightGCN, ItemKNN, NeuMF
from recbole.model.context_aware_recommender import FM, DeepFM, WideDeep
from recbole.model.knowledge_aware_recommender import KGCN, KGIN, KGAT

from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_trainer

from models import MODELS
from utils import load_config, ROOT_PATH

from pathlib import Path

import warnings

# Suppress specific warning types
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def custom_objective():
    pass

def write_yaml(file_path, data):
    """
    Write data to a YAML file.

    Args:
        file_path (str): Path to the YAML file.
        data (dict): Data to write to the file.
    """
    import yaml
    if not Path(file_path).is_absolute():
        file_path = ROOT_PATH / file_path
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def hyperparameter_tuning(model_name, dataset):
    base_config = load_config(ROOT_PATH / 'config' / 'config.yaml')
    write_yaml(ROOT_PATH / 'config' / 'temp.config', base_config)


    print('config_dict: ', base_config)
    hp = HyperTuning(
        objective_function=objective_function,
        algo='exhaustive',
        params_file=ROOT_PATH / 'config' / 'model.hyper',
        fixed_config_file_list= [ROOT_PATH / 'config' / 'temp.config']
    )

        # Inject model and dataset
    # base_config['model'] = model_name
    # base_config['dataset'] = dataset

    # def wrapped_objective_function(config_updates):
    #     # Merge current trial config with base
    #     full_config = {**base_config, **config_updates}
    #     return objective_function(full_config)

    # hp = HyperTuning(
    #     objective_function=wrapped_objective_function,
    #     algo='exhaustive',
    #     params_file=ROOT_PATH / 'config' / 'model.hyper',
    # )


    hp.run()
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])
    # Update config with best hyperparameters
    config = Config(model=model_name, dataset=dataset, config_dict=hp.best_params)
    return config


def train_test(dataset_name='ml-100k'):
    print("Training and Testing with dataset:", dataset_name)
    config = Config(config_dict=load_config('config/config.yaml'))
    config['dataset'] = dataset_name
    print("Configuration loaded:", config)

    # Initialization
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()

    print("Initializing dataset...")
    # Prepare dataset [TODO: CHANGE TO CUSTOM DATASET]
    dataset = create_dataset(config)
    logger.info("Dataset: {}".format(dataset))

    for model_name, model_class in MODELS.items():
        print(f"Training model: {model_name}")
        config['model'] = model_name
        config = hyperparameter_tuning(model_name, dataset_name)
        tr, va, te = data_preparation(config, dataset)
        # Initialize model
        model = model_class(config, tr._dataset).to(config['device'])
    
        # Initialize trainer
        trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
        trainer = trainer_class(config, model)
        best_valid_score, best_valid_result = trainer.fit(tr, va)

        # Evaluation
        test_result = trainer.evaluate(te)
        print(test_result)


if __name__ == "__main__":
    train_test('ml-100k')
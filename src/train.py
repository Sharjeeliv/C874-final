# *****************************
# IMPORTS
# *****************************
import logging
from logging import getLogger
from recbole.config import Config

from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_trainer

# Suppressing warnings: Recbole-inherited warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# local Imports
from models import MODELS
from utils import load_config, write_yaml, del_yaml, del_dir, print_results, ROOT_PATH

# *****************************
#  CONST & SETUP
# *****************************
MAX_EVALS =  50
EARLY_STOP = 10



# *****************************
# HELPER FUNCTIONS
# *****************************
def hyperparameter_tuning(model_name, dataset):
    # Dynamically load the base configuration
    base_config = load_config(ROOT_PATH / 'config' / 'config.yaml')
    base_config['model'] = model_name
    base_config['dataset'] = dataset
    temp_config = ROOT_PATH / 'config' / 'temp.yaml'
    write_yaml(temp_config, base_config)

    # Load the model-specific hyperparameters
    param_file = ROOT_PATH / 'config' / 'model.hyper'
    hp = HyperTuning(
        objective_function=objective_function,
        algo='random',
        params_file=param_file,
        max_evals=MAX_EVALS,
        early_stop=EARLY_STOP,
        fixed_config_file_list=[temp_config]
    )
    # Run hyperparameter tuning
    hp.run()
    print('Best Params:', hp.best_params)
    print('Best Result:')
    # Save parameters to auto load from file
    param_dir = ROOT_PATH / 'config' / 'params'
    param_dir.mkdir(exist_ok=True)
    path = param_dir / f"{model_name}_{dataset}_params.yaml"
    write_yaml(path, hp.best_params)
    print(hp.params2result[hp.params2str(hp.best_params)])
    # Clean up temporary config file and return
    del_yaml(temp_config)
    merged_config = {**base_config, **hp.best_params}
    merged_config['state'] = str(base_config['state'])
    config = Config(model=model_name, dataset=dataset, config_dict=merged_config)
    return config


# *****************************
# MAIN TRAIN-TEST FUNCTION
# *****************************
def train_test(dataset_name='ml-100k', tune=True, use_100k=False):

    results_dir = ROOT_PATH / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"\033[93mDataset: {dataset_name}\033[0m")

    for model_name, model_class in MODELS.items():
        print(f"\033[93mModel: {model_name}\033[0m")

        # Step 1: Tune and get Config
        if tune:
            print("\033[92mTuning hyperparameters...\033[0m")
            config = hyperparameter_tuning(model_name, dataset_name)
        else:
            param_dir = ROOT_PATH / 'config' / 'params'
            param_file = param_dir / f"{model_name}_{dataset_name}_params.yaml"
            if use_100k: 
                param_file = param_dir / f"{model_name}_ml-100k_params.yaml"
            if not param_file.exists(): 
                raise FileNotFoundError(f"Parameter file not found: {param_file}")
            print("\033[92mLoading hyperparameters...\033[0m")
            base_config = load_config(ROOT_PATH / 'config' / 'config.yaml')
            config = {**base_config, **load_config(param_file)}
            config = Config(model=model_name, dataset=dataset_name, config_dict=config)

        print(f"\033[92mTuned Config: {config}\033[0m")
        # Step 2: Init seed/logger
        init_seed(config['seed'], config['reproducibility'])

        # Step 3: Prepare dataset using the tuned config
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # Step 4: Model initialization
        model = model_class(config, train_data._dataset).to(config['device'])

        # Step 5: Trainer
        trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
        trainer = trainer_class(config, model)
        bvs, bvr = trainer.fit(train_data, valid_data)

        # Step 6: Evaluation
        result = trainer.evaluate(test_data)
        result_file = results_dir / f"{model_name}_{dataset_name}_result.yaml"
        print_results(result, result_file)

    # Clear logs
    del_dir('saved')
    del_dir('log_tensorboard')


# *****************************
# MAIN FUNCTION
# *****************************
if __name__ == "__main__":
    train_test('ml-1m', tune=False, use_100k=True)

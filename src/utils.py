from pathlib import Path
import yaml

ROOT_PATH = Path(__file__).resolve().parent.parent

def load_config(file_path):
    """
    Load a YAML configuration file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Configuration parameters as a dictionary.
    """
    if not Path(file_path).is_absolute():
        file_path = ROOT_PATH / file_path

    config = open(file_path, 'r').read()
    config = config.replace('ROOT_PATH', str(ROOT_PATH))
    config = config.replace('\\', '/')
    config = yaml.safe_load(config)
    return config

if __name__ == "__main__":
    print(ROOT_PATH)
    config = load_config(r'C:\Users\Sharjeel Mustafa\Documents\Documents\Academic\C874\C874-final-project\config\config.yaml')
    print(config)
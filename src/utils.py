from pathlib import Path
import yaml

ROOT_PATH = Path(__file__).resolve().parent.parent


# *****************************
# HELPER FUNCTIONS
# *****************************
def print_results(results, output_file):
    if not Path(output_file).is_absolute():
        output_file = ROOT_PATH / output_file
    with open(output_file, 'w') as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")
    print(f"Results saved to {output_file}")


def del_dir(directory):
    if not Path(directory).is_absolute():
        directory = ROOT_PATH / directory
    if Path(directory).exists():
        for item in directory.iterdir():
            if item.is_dir(): del_dir(item)
            else: item.unlink()
        directory.rmdir()


# *****************************
# CONFIG HELPER FUNCTIONS
# *****************************
def write_yaml(file_path, data):
    if not Path(file_path).is_absolute():
        file_path = ROOT_PATH / file_path
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def del_yaml(file_path):
    if not Path(file_path).is_absolute():
        file_path = ROOT_PATH / file_path
    if Path(file_path).exists():
        Path(file_path).unlink()

def load_config(file_path):
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
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

def format_results(dataset_name):
    result_path = ROOT_PATH / "results"
    for file in result_path.glob(f"*{dataset_name}_*.yaml"):
        with open(file, 'r') as f: results = yaml.safe_load(f)
        # Print out horizontal table with results
        print(f"\033[93mResults for {file.stem}:\033[0m")
        print("-" * 40)
        # Print out each metric in a formatted way
        print(f"\nResults for {file.stem}:")
        print("-" * 40)
        # Include all metrics
        for key, value in results.items():
            if key.startswith('recall@') or key.startswith('precision@') or key.startswith('ndcg@') or key.startswith('hit@'):
                if isinstance(value, float):
                    value = f"{value:.4f}"
                
            # Align the output
            if isinstance(value, list):
                value = ', '.join(f"{v:.4f}" for v in value)
            print(f"{key:<20}: {value}")
        print("-" * 40)


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
    format_results('ml-100k')
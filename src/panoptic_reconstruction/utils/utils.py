import os
import yaml
import shutil
import pandas as pd
from typing import Dict, Any


def get_relative_path(x: str, rel_to: str) -> str:
    """Constructs a relative path by combining the directory of `rel_to` with `x`."""
    return os.path.join(os.path.dirname(rel_to), x)


def load_yaml(x: str) -> Dict[str, Any]:
    """Loads a YAML file and parses its contents into a Python dictionary."""
    with open(x) as fd:
        config = yaml.load(fd, yaml.FullLoader)
        return config


def delete_auxiliar_data(save_origin_dir: str, output_cropped: str) -> None:
    """Deletes auxiliary data directories if they exist."""
    if os.path.exists(save_origin_dir):
        shutil.rmtree(save_origin_dir)

    if os.path.exists(output_cropped):
        shutil.rmtree(output_cropped)


def read_df_file(filename: str) -> pd.DataFrame:
    """Reads a CSV file with semicolon delimiter containing paths to train/valid/test images."""
    return pd.read_csv(filename, delimiter=';')

"""
Adapted from https://github.com/guillaume-be/rust-bert/blob/master/utils/convert_model.py
"""

from pathlib import Path
import numpy as np
import torch
import subprocess
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Utility script to convert a pytorch transformer model to a format that can be loaded in libtorch/tch")
    parser.add_argument("--model_path", help="Path to pytorch model")
    parser.add_argument(
        "--output_path", help="Folder path to output the converted weights")
    args = parser.parse_args()

    model_path, output_path = args.model_path, args.output_path

    assert os.path.exists(model_path), "Model path is not a valid path!"
    os.makedirs(output_path, exist_ok=True)

    weights = torch.load(model_path, map_location='cpu')['model'].state_dict()

    np_weights = {}
    for k, v in weights.items():
        k = k.replace("gamma", "weight").replace("beta", "bias")
        np_weights[k] = np.ascontiguousarray(v.numpy())

    npz_path = os.path.join(output_path, 'model.npz')
    np.savez(npz_path, **np_weights)
    new_basename = os.path.basename(model_path).rsplit('.', 1)[0] + ".ot"

    rust_model_path = os.path.join(output_path, new_basename)
    toml_location = (Path(__file__).resolve() / '..' /
                     '..' / 'Cargo.toml').resolve()
    subprocess.run(
        ['cargo', 'run', '--bin=convert_tensors', '--manifest-path=%s' %
            toml_location, '--', "-p", npz_path, "-r", rust_model_path],
    )

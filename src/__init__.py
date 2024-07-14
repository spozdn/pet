from .single_struct_calculator import SingleStructCalculator

import torch
import importlib.resources as pkg_resources

def load_neighbors_convert():
    try:
        # Locate the shared object file in the package
        with pkg_resources.files(__name__).joinpath('neighbors_convert.so') as lib_path:
            # Load the shared object file
            torch.ops.load_library(str(lib_path))
    except Exception as e:
        print(f"Failed to load neighbors_convert.so: {e}")

load_neighbors_convert()

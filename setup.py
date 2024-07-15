from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, include_paths, library_paths

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Collecting include and library paths
include_dirs = include_paths()
library_dirs = library_paths()

libraries = []

libraries.append('c10')
libraries.append('torch')
libraries.append('torch_cpu')


# Defining the extension module without specifying the unwanted libraries
neighbors_convert_extension = Extension(
    name="pet.neighbors_convert",
    sources=["src/neighbors_convert.cpp"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    language='c++',
)

setup(
    name="pet",
    version="0.0.0",
    packages=["pet"],
    package_dir={"pet": "src"},
    entry_points={
        "console_scripts": [
            "pet_train = pet.train_model:main",
            "pet_run = pet.estimate_error:main",
            "pet_run_sp = pet.estimate_error_sp:main",
            "pet_train_general_target = pet.train_model_general_target:main",
        ],
    },
    install_requires=requirements,
    ext_modules=[neighbors_convert_extension],
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True)
    },
    package_data={
        'pet': ['neighbors_convert.so'],  # Ensure the shared object file is included
    },
    include_package_data=True,
)

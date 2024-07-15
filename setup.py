from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

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
    ext_modules=[
        CppExtension(
            name="pet.neighbors_convert",  # Ensure this matches the package structure
            sources=["src/neighbors_convert.cpp"],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True)
    },
    package_data={
        'pet': ['neighbors_convert.so'],  # Ensure the shared object file is included
    },
    include_package_data=True,
)

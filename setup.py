import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

extra_compile_args = []
extra_link_args = []

if sys.platform == "darwin":
    # macOS specific flags for OpenMP
    extra_compile_args = ['-Xpreprocessor', '-fopenmp']
    extra_link_args = ['-lomp']
else:
    # General flags for other platforms
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']

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
            name="neighbors_convert",
            sources=["src/neighbors_convert.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)

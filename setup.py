from setuptools import setup


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
)

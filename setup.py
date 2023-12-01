from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pet',
    version='0.0.0',
    packages=['pet'],
    package_dir={'pet':'src'},
    entry_points={
        'console_scripts': [
            'pet_train_model = pet.train_model:main',
            'pet_estimate_error = pet.estimate_error:main',
            'pet_estimate_error_sp = pet.estimate_error_sp:main'
        ],
    },
    install_requires=requirements,  
)
    
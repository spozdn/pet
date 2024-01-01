import subprocess
import pytest
import shutil
import os
from pet import SingleStructCalculator
import ase.io


def clean():
    """
    Clean the outputs potentially remaining from the previous run.

    This function removes the 'results' directory which may contain
    data from the previous run, ensuring a clean state for the current run.
    """
    results_dir = "results"
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)


@pytest.fixture
def prepare_model():
    """
    Prepare a model for testing 'pet_run' and 'single_struct_calculator'.

    Returns:
        model_folder (str): Path to the model folder.
    """
    clean()

    script = "pet_train"
    args = [
        "../example/methane_train.xyz",
        "../example/methane_val.xyz",
        "hypers_minimal.yaml",
        "../default_hypers/default_hypers.yaml",
        "test",
    ]

    process = subprocess.run(
        [script] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, "pet_train script failed"

    model_folder = "results/test"
    assert os.path.exists(
        model_folder
    ), "pet_train script failed to create the model folder"
    return model_folder


def test_pet_train():
    """
    Test the 'pet_train' script for successful execution.

    This test runs the 'pet_train' script and asserts that it completes
    successfully (return code 0).
    """
    clean()

    script = "pet_train"
    args = [
        "../example/methane_train.xyz",
        "../example/methane_val.xyz",
        "hypers_minimal.yaml",
        "../default_hypers/default_hypers.yaml",
        "test",
    ]

    process = subprocess.run(
        [script] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, "pet_train script failed"


def test_pet_run(prepare_model):
    """
    Test the 'pet_run' script using the model prepared by 'prepare_model'.

    This test runs the 'pet_run' script. It asserts that the
    script completes successfully.
    """
    model_folder = prepare_model
    script = "pet_run"

    args = [
        "../example/methane_test.xyz",
        model_folder,
        "best_val_rmse_both_model",
        "1",
        "100",
    ]

    process = subprocess.run(
        [script] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, "pet_run script failed"


def test_single_struct_calculator(prepare_model):
    """
    Test the SingleStructCalculator class with a prepared model.

    This test initializes SingleStructCalculator with a model
    provided by 'prepare_model'. It then computes energy and forces
    on a test structure and checks that the output forces have the correct shape.
    """
    model_folder = prepare_model
    single_struct_calculator = SingleStructCalculator(
        model_folder,
    )
    structure = ase.io.read("../example/methane_test.xyz", index=0)
    energy, forces = single_struct_calculator.forward(structure)

    assert forces.shape == (5, 3), "single_struct_calculator failed"


@pytest.fixture(scope="session", autouse=True)
def run_at_the_end(request):
    """
    Register a finalizer to clean the temporarily files
    at the end of the test session.
    """
    request.addfinalizer(clean)

import subprocess
import pytest


def test_pet_train():
    command = (
        "pet_train ../example/methane_train.xyz ../example/methane_val.xyz "
        "hypers_minimal.yaml ../default_hypers/default_hypers.yaml test"
    )
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.returncode == 0, "pet_train script failed"

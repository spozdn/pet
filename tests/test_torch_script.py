import pytest
import torch

from pet.pet import PET
from pet.hypers import load_hypers_from_file


def test_torch_scriptability():
    """Test that the model is scriptable"""
    hypers = load_hypers_from_file("../default_hypers/default_hypers.yaml")
    ARCHITECTURAL_HYPERS = hypers.ARCHITECTURAL_HYPERS
    ARCHITECTURAL_HYPERS.D_OUTPUT = 1
    ARCHITECTURAL_HYPERS.TARGET_TYPE = "structural"
    ARCHITECTURAL_HYPERS.TARGET_AGGREGATION = "sum"

    boolean_architectural_hypers = [
        "AVERAGE_POOLING",
        "ADD_TOKEN_FIRST",
        "ADD_TOKEN_SECOND",
        "R_EMBEDDING_ACTIVATION",
        "BLEND_NEIGHBOR_SPECIES",
        "AVERAGE_BOND_ENERGIES",
        "USE_ONLY_LENGTH",
        "USE_LENGTH",
    ]

    for key in boolean_architectural_hypers:
        initial_value = ARCHITECTURAL_HYPERS.__dict__[key]
        ARCHITECTURAL_HYPERS.__dict__[key] = True
        model = PET(ARCHITECTURAL_HYPERS, 0.0, 1)
        model = torch.jit.script(model)

        ARCHITECTURAL_HYPERS.__dict__[key] = False
        model = PET(ARCHITECTURAL_HYPERS, 0.0, 1)
        model = torch.jit.script(model)

        ARCHITECTURAL_HYPERS.__dict__[key] = initial_value

    ARCHITECTURAL_HYPERS.SCALAR_ATTRIBUTES_SIZE = 15
    ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES = True
    model = PET(ARCHITECTURAL_HYPERS, 0.0, 1)
    model = torch.jit.script(model)

    assert True

import pytest
from itertools import product

import ase.io
from pet.hypers import load_hypers_from_file
from pet.data_preparation import get_all_species
from pet.pet import PET, PETUtilityWrapper, PETMLIPWrapper
import torch
from pet.molecule import MoleculeCPP, Molecule
from matscipy.neighbours import neighbour_list as neighbor_list

def prepare_test(stucture_path, r_cut, n_gnn, n_trans, structure_index, hypers_path = "../default_hypers/default_hypers.yaml"):
    device = 'cpu'
    structure = ase.io.read(stucture_path, index=structure_index)
    hypers = load_hypers_from_file(hypers_path)


    MLIP_SETTINGS = hypers.MLIP_SETTINGS
    ARCHITECTURAL_HYPERS = hypers.ARCHITECTURAL_HYPERS
    FITTING_SCHEME = hypers.FITTING_SCHEME

    ARCHITECTURAL_HYPERS.D_OUTPUT = 1  # energy is a single scalar
    ARCHITECTURAL_HYPERS.TARGET_TYPE = "structural"  # energy is structural property
    ARCHITECTURAL_HYPERS.TARGET_AGGREGATION = (
        "sum"  # energy is a sum of atomic energies
    )
    ARCHITECTURAL_HYPERS.R_CUT = r_cut
    ARCHITECTURAL_HYPERS.N_TRANS_LAYERS = n_trans
    ARCHITECTURAL_HYPERS.N_GNN_LAYERS = n_gnn
    all_species = get_all_species([structure])


    model = PET(ARCHITECTURAL_HYPERS, 0.0, len(all_species)).to(device)
    model = PETUtilityWrapper(model, FITTING_SCHEME.GLOBAL_AUG)

    model = PETMLIPWrapper(
        model, MLIP_SETTINGS.USE_ENERGIES, MLIP_SETTINGS.USE_FORCES
    )
    return model, structure, all_species, ARCHITECTURAL_HYPERS

def get_predictions_old_python(model, structure, all_species, ARCHITECTURAL_HYPERS):
    device = 'cpu'
    molecule = Molecule(
        structure,
        ARCHITECTURAL_HYPERS.R_CUT,
        ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
        ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
        ARCHITECTURAL_HYPERS.K_CUT,
        ARCHITECTURAL_HYPERS.N_TARGETS > 1,
        ARCHITECTURAL_HYPERS.TARGET_INDEX_KEY
    )
    if ARCHITECTURAL_HYPERS.USE_LONG_RANGE:
        raise NotImplementedError(
            "Long range interactions are not supported in the SingleStructCalculator"
        )

    graph = molecule.get_graph(
        molecule.get_max_num(), all_species, None
    )
    graph.batch = torch.zeros(
        graph.num_nodes, dtype=torch.long, device=graph.x.device
    )
    graph = graph.to(device)
    prediction_energy, prediction_forces = model(
        graph, augmentation=False, create_graph=False
    )

    return prediction_energy, prediction_forces, graph

def get_predictions_cpp(model, structure, all_species, ARCHITECTURAL_HYPERS):
    device = 'cpu'
    molecule = MoleculeCPP(
        structure,
        ARCHITECTURAL_HYPERS.R_CUT,
        ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
        ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
        ARCHITECTURAL_HYPERS.K_CUT,
        ARCHITECTURAL_HYPERS.N_TARGETS > 1,
        ARCHITECTURAL_HYPERS.TARGET_INDEX_KEY
    )
    if ARCHITECTURAL_HYPERS.USE_LONG_RANGE:
        raise NotImplementedError(
            "Long range interactions are not supported in the SingleStructCalculator"
        )

    graph = molecule.get_graph(
        molecule.get_max_num(), all_species, None
    )
    graph.batch = torch.zeros(
        graph.num_nodes, dtype=torch.long, device=graph.x.device
    )
    graph = graph.to(device)
    prediction_energy, prediction_forces = model(
        graph, augmentation=False, create_graph=False
    )

    return prediction_energy, prediction_forces, graph

class Float64DtypeContext:
    def __enter__(self):
        # Save the current default dtype
        self.original_dtype = torch.get_default_dtype()
        # Set the default dtype to float64
        torch.set_default_dtype(torch.float64)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the original default dtype
        torch.set_default_dtype(self.original_dtype)


def do_single_test(stucture_path, r_cut, n_gnn, n_trans, structure_index, epsilon):
    model, structure, all_species, ARCHITECTURAL_HYPERS = prepare_test(stucture_path, r_cut, n_gnn, n_trans, structure_index)
    python_energy, python_forces, python_graph = get_predictions_old_python(model, structure, all_species, ARCHITECTURAL_HYPERS)
    cpp_energy, cpp_forces, cpp_graph = get_predictions_cpp(model, structure, all_species, ARCHITECTURAL_HYPERS)
    # print(f"energy difference: {torch.abs(python_energy - cpp_energy)}")
    # print(f"forces difference: {torch.abs(python_forces - cpp_forces).max()}")
    assert torch.abs(python_energy - cpp_energy) < epsilon, f"Energy difference is {torch.abs(python_energy - cpp_energy)}"
    assert torch.abs(python_forces - cpp_forces).max() < epsilon, f"Max force difference is {torch.abs(python_forces - cpp_forces).max()}"



# Define the parameters for each case
case1_params = ("../example/methane_train.xyz", 0, [1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
case2_params = ("bulk.xyz", 0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
case3_params = ("bulk_small_unit_cell.xyz", list(range(9)), [2.0, 3.0, 5.0, 10.0, 15.0, 20.0])

# Generate the expanded lists using product
expanded_case1 = list(product([case1_params[0]], [case1_params[1]], case1_params[2]))
expanded_case2 = list(product([case2_params[0]], [case2_params[1]], case2_params[2]))
expanded_case3 = list(product([case3_params[0]], case3_params[1], case3_params[2]))

# Combine all cases into one list
all_cases = expanded_case1 + expanded_case2 + expanded_case3

@pytest.mark.parametrize("structures_path, structure_index, r_cut", all_cases)
def test_do_single(structures_path, structure_index, r_cut):
    n_gnn = 2
    n_trans = 2
    epsilon = 1e-10
    with Float64DtypeContext():
        do_single_test(structures_path, r_cut, n_gnn, n_trans, structure_index, epsilon)
    #do_single_test(structures_path, r_cut, n_gnn, n_trans, structure_index, epsilon)
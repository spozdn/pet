from .utilities import get_compositional_features

import torch
import ase.io
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import sys
from torch_geometric.nn import DataParallel

from .molecule import Molecule
from .hypers import Hypers
from .pet import PET, PETUtilityWrapper
from .data_preparation import get_rmse, get_mae


from sp_frames_calculator import SPFramesCalculator
from pet_sp import PETSP


def main():
    np.random.seed(0)

    EPSILON = 1e-10

    STRUCTURES_PATH = sys.argv[1]
    PATH_TO_CALC_FOLDER_MAIN = sys.argv[2]
    CHECKPOINT_MAIN = sys.argv[3]

    PATH_TO_CALC_FOLDER_AUX = sys.argv[4]
    CHECKPOINT_AUX = sys.argv[5]

    bool_map = {"True": True, "False": False}

    SP_HYPERS_PATH = sys.argv[6]
    DEFAULT_HYPERS_PATH = sys.argv[7]

    BATCH_SIZE_SP = int(sys.argv[8])
    PATH_SAVE_PREDICTIONS = sys.argv[9]
    SHOW_PROGRESS = bool_map[sys.argv[10]]
    MAX_NUM = sys.argv[11]

    if MAX_NUM == "None" or MAX_NUM == "none":
        MAX_NUM = None
    else:
        MAX_NUM = int(sys.argv[11])

    def load_model(path_to_calc_folder, checkpoint):
        hypers_path = path_to_calc_folder + "/hypers_used.yaml"
        path_to_model_state_dict = (
            path_to_calc_folder + "/" + checkpoint + "_state_dict"
        )
        all_species_path = path_to_calc_folder + "/all_species.npy"
        self_contributions_path = path_to_calc_folder + "/self_contributions.npy"

        hypers = Hypers()

        # loading default values for the new hypers potentially added into the codebase after the calculation is done
        # assuming that the default values do not change the logic
        hypers.set_from_files(hypers_path, DEFAULT_HYPERS_PATH, check_duplicated=False)

        all_species = np.load(all_species_path)
        if hypers.USE_ENERGIES:
            self_contributions = np.load(self_contributions_path)
        else:
            self_contributions = None

        add_tokens = []
        for _ in range(hypers.N_GNN_LAYERS - 1):
            add_tokens.append(hypers.ADD_TOKEN_FIRST)
        add_tokens.append(hypers.ADD_TOKEN_SECOND)

        model = PET(
            hypers,
            hypers.TRANSFORMER_D_MODEL,
            hypers.TRANSFORMER_N_HEAD,
            hypers.TRANSFORMER_DIM_FEEDFORWARD,
            hypers.N_TRANS_LAYERS,
            0.0,
            len(all_species),
            hypers.N_GNN_LAYERS,
            hypers.HEAD_N_NEURONS,
            hypers.TRANSFORMERS_CENTRAL_SPECIFIC,
            hypers.HEADS_CENTRAL_SPECIFIC,
            add_tokens,
        ).cuda()
        model = PETUtilityWrapper(model)

        if hypers.MULTI_GPU:
            model = DataParallel(model)
            device = torch.device("cuda:0")
            model = model.to(device)

            model.load_state_dict(torch.load(path_to_model_state_dict))
            model.eval()

            model = model.module
        else:
            model.load_state_dict(torch.load(path_to_model_state_dict))
            model.eval()

        return model, hypers, all_species, self_contributions

    model_main, hypers_main, all_species_main, self_contributions_main = load_model(
        PATH_TO_CALC_FOLDER_MAIN, CHECKPOINT_MAIN
    )

    def are_same(first, second):
        if len(first) != len(second):
            return False

        for i in range(len(first)):
            if abs(first[i] - second[i]) > EPSILON:
                return False
        return True

    if PATH_TO_CALC_FOLDER_AUX == "None" or PATH_TO_CALC_FOLDER_AUX == "none":
        PATH_TO_CALC_FOLDER_AUX = None

    if PATH_TO_CALC_FOLDER_AUX is None:
        model_aux, hypers_aux, all_species_aux, self_contributions_aux = (
            None,
            None,
            None,
            None,
        )

        USE_ADDITIONAL_SCALAR_ATTRIBUTES_DATA = (
            hypers_main.USE_ADDITIONAL_SCALAR_ATTRIBUTES
        )
        USE_FORCES = hypers_main.USE_FORCES
        USE_ENERGIES = hypers_main.USE_ENERGIES

        if hypers_main.MULTI_GPU:
            print("using only 1 gpu, multigpu for sp is not implemented")

    else:
        model_aux, hypers_aux, all_species_aux, self_contributions_aux = load_model(
            PATH_TO_CALC_FOLDER_AUX, CHECKPOINT_AUX
        )

        print("self contributions main", self_contributions_main)
        print("self contributions aux", self_contributions_aux)
        print(
            "self contributions delta", self_contributions_main - self_contributions_aux
        )

        if np.abs(hypers_aux.R_CUT - hypers_main.R_CUT) > EPSILON:
            raise ValueError(
                "R_CUT of main and aux models should be same in the current implementation"
            )

        if not are_same(self_contributions_main, self_contributions_aux):
            raise ValueError(
                "self contributions should be same (in this rudementary implementation)"
            )

        if not are_same(all_species_main, all_species_aux):
            raise ValueError("all species should be same")

        USE_ADDITIONAL_SCALAR_ATTRIBUTES_DATA = (
            hypers_main.USE_ADDITIONAL_SCALAR_ATTRIBUTES
            or hypers_aux.USE_ADDITIONAL_SCALAR_ATTRIBUTES
        )

        USE_FORCES = hypers_main.USE_FORCES and hypers_aux.USE_FORCES
        USE_ENERGIES = hypers_main.USE_ENERGIES and hypers_aux.USE_ENERGIES

        if hypers_main.MULTI_GPU or hypers_aux.MULTI_GPU:
            print("using only 1 gpu, multigpu for sp is not implemented")

    R_CUT = hypers_main.R_CUT
    self_contributions = self_contributions_main
    all_species = all_species_main

    structures = ase.io.read(STRUCTURES_PATH, index=":")

    molecules = [
        Molecule(
            structure,
            R_CUT,
            USE_ADDITIONAL_SCALAR_ATTRIBUTES_DATA,
            USE_FORCES,
            hypers_main.FORCES_KEY,
        )
        for structure in tqdm(structures)
    ]
    max_nums = [molecule.get_max_num() for molecule in molecules]
    max_num = np.max(max_nums)
    graphs = [molecule.get_graph(max_num, all_species) for molecule in tqdm(molecules)]

    loader = DataLoader(graphs, 1, shuffle=False)

    if USE_ENERGIES:
        energies_ground_truth = np.array(
            [struc.info[hypers_main.ENERGY_KEY] for struc in structures]
        )

    if USE_FORCES:
        forces_ground_truth = [
            struc.arrays[hypers_main.FORCES_KEY] for struc in structures
        ]
        forces_ground_truth = np.concatenate(forces_ground_truth, axis=0)

    sp_hypers = Hypers()
    sp_hypers.load_from_file(SP_HYPERS_PATH)
    sp_frames_calculator = SPFramesCalculator(sp_hypers)

    model_sp = PETSP(
        model_main,
        model_aux,
        R_CUT,
        USE_ENERGIES,
        USE_FORCES,
        sp_frames_calculator,
        BATCH_SIZE_SP,
        len(all_species),
        epsilon=EPSILON,
        show_progress=SHOW_PROGRESS,
        max_num=MAX_NUM,
        n_aug=sp_hypers.N_ADDITIONAL_AUG,
    ).cuda()

    if USE_ENERGIES:
        all_energies_predicted = []

    if USE_FORCES:
        all_forces_predicted = []

    if USE_ENERGIES:
        energies_predicted = []
    if USE_FORCES:
        forces_predicted = []

    # print("len loader: ", len(loader), len(molecules))

    n_frames_used, aux_weights, total_main_weights = [], [], []
    for batch in tqdm(loader):
        # print(batch)
        batch.cuda()
        # with torch.autograd.set_detect_anomaly(True):
        (
            n_frames,
            aux_weight,
            total_main_weight,
            predictions_energies,
            targets_energies,
            predictions_forces,
            targets_forces,
        ) = model_sp(batch)
        n_frames_used.append(n_frames)
        if isinstance(total_main_weight, float):
            total_main_weights.append(total_main_weight)
        else:
            total_main_weights.append(total_main_weight.data.cpu().numpy())

        aux_weights.append(float(aux_weight.data.cpu().numpy()))
        if USE_ENERGIES:
            energies_predicted.append(predictions_energies.data.cpu().numpy())
        if USE_FORCES:
            forces_predicted.append(predictions_forces.data.cpu().numpy())

    if USE_ENERGIES:
        energies_predicted = np.concatenate(energies_predicted, axis=0)
        all_energies_predicted.append(energies_predicted)

    if USE_FORCES:
        forces_predicted = np.concatenate(forces_predicted, axis=0)
        all_forces_predicted.append(forces_predicted)

    if USE_ENERGIES:
        all_energies_predicted = [el[np.newaxis] for el in all_energies_predicted]
        all_energies_predicted = np.concatenate(all_energies_predicted, axis=0)
        energies_predicted_mean = np.mean(all_energies_predicted, axis=0)

    if USE_FORCES:
        all_forces_predicted = [el[np.newaxis] for el in all_forces_predicted]
        all_forces_predicted = np.concatenate(all_forces_predicted, axis=0)
        forces_predicted_mean = np.mean(all_forces_predicted, axis=0)

    print("Average number of active coordinate systems: ", np.mean(n_frames_used))
    # print("aux_weights: ", aux_weights)
    n_fully_aux, n_partially_aux = 0, 0
    for weight in aux_weights:
        if weight > EPSILON:
            n_partially_aux += 1

    # print(total_main_weights)
    for weight in total_main_weights:
        if weight < EPSILON:
            n_fully_aux += 1

    # print("The number of structures handled completely by auxiliary model is: ", n_fully_aux, '; ratio is', n_fully_aux / len(aux_weights))
    if PATH_TO_CALC_FOLDER_AUX is not None:
        print(
            f"Auxiliary model was active for {n_partially_aux}/{len(aux_weights)} structures "
        )

    if USE_ENERGIES:
        compositional_features = get_compositional_features(structures, all_species)
        self_contributions_energies = []
        for i in range(len(structures)):
            self_contributions_energies.append(
                np.dot(compositional_features[i], self_contributions)
            )
        self_contributions_energies = np.array(self_contributions_energies)

        energies_predicted_mean = energies_predicted_mean + self_contributions_energies

        print(
            f"energies mae: {get_mae(energies_ground_truth, energies_predicted_mean)}"
        )
        print(
            f"energies rmse: {get_rmse(energies_ground_truth, energies_predicted_mean)}"
        )

    if USE_FORCES:
        print(
            f"forces mae per component: {get_mae(forces_ground_truth, forces_predicted_mean)}"
        )
        print(
            f"forces rmse per component: {get_rmse(forces_ground_truth, forces_predicted_mean)}"
        )

    if (PATH_SAVE_PREDICTIONS != "None") and (PATH_SAVE_PREDICTIONS != "none"):
        if USE_ENERGIES:
            np.save(
                PATH_SAVE_PREDICTIONS + "/energies_predicted.npy",
                energies_predicted_mean,
            )
        if USE_FORCES:
            np.save(
                PATH_SAVE_PREDICTIONS + "/forces_predicted.npy", forces_predicted_mean
            )


if __name__ == "__main__":
    main()

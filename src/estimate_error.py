

import torch
import ase.io
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader, DataListLoader
import time
from torch_geometric.nn import DataParallel


from .hypers import Hypers
from .pet import PET
from .utilities import get_rmse, get_mae, set_reproducibility
import argparse
from .data_preparation import get_pyg_graphs, get_compositional_features
from .data_preparation import update_pyg_graphs, get_forces

def main():
    parser = argparse.ArgumentParser()


    parser.add_argument("structures_path", help="Path to an xyz file with structures", type = str)
    parser.add_argument("path_to_calc_folder", help="Path to a folder with a model to use", type = str)
    parser.add_argument("checkpoint", help="Path to a particular checkpoint to use", type = str, choices = ['best_val_mae_energies_model', 'best_val_rmse_energies_model', 'best_val_mae_forces_model', 'best_val_rmse_forces_model',  'best_val_mae_both_model', 'best_val_rmse_both_model'])

    parser.add_argument("n_aug", type = int, help = "A number of rotational augmentations to use. It should be a positive integer or -1. If -1, the initial coordinate system will be used, not a single random one, as in the n_aug = 1 case")
    parser.add_argument("default_hypers_path", help="Path to a YAML file with default hypers", type = str)

    parser.add_argument("batch_size", type = int, help="Batch size to use for inference. It should be a positive integer or -1. If -1, it will be set to the value used for fitting the provided model.")

    parser.add_argument("--path_save_predictions", help="Path to a folder where to save predictions.", type = str)
    parser.add_argument("--verbose", help="Show more details",
                        action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    HYPERS_PATH = args.path_to_calc_folder + '/hypers_used.yaml'
    PATH_TO_MODEL_STATE_DICT = args.path_to_calc_folder + '/' + args.checkpoint + '_state_dict'
    ALL_SPECIES_PATH = args.path_to_calc_folder + '/all_species.npy'
    SELF_CONTRIBUTIONS_PATH = args.path_to_calc_folder + '/self_contributions.npy'

    if args.n_aug == -1:
        N_AUG = 1
        USE_AUGMENTATION = False
    else:
        N_AUG = args.n_aug
        USE_AUGMENTATION = True

    hypers = Hypers()
    # loading default values for the new hypers potentially added into the codebase after the calculation is done
    # assuming that the default values do not change the logic
    hypers.set_from_files(HYPERS_PATH, args.default_hypers_path, check_duplicated = False)

    set_reproducibility(hypers.RANDOM_SEED, hypers.CUDA_DETERMINISTIC)

    if args.batch_size == -1:
        args.batch_size = hypers.STRUCTURAL_BATCH_SIZE

    structures = ase.io.read(args.structures_path, index = ':')

    all_species = np.load(ALL_SPECIES_PATH)
    if hypers.USE_ENERGIES:
        self_contributions = np.load(SELF_CONTRIBUTIONS_PATH)

    graphs = get_pyg_graphs(structures, all_species, hypers.R_CUT, hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES)
    forces = get_forces(structures, hypers.FORCES_KEY)
    update_pyg_graphs(graphs, 'forces', forces)

    if hypers.MULTI_GPU:
        loader = DataListLoader(graphs, batch_size=args.batch_size, shuffle=False)
    else:        
        loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    add_tokens = []
    for _ in range(hypers.N_GNN_LAYERS - 1):
        add_tokens.append(hypers.ADD_TOKEN_FIRST)
    add_tokens.append(hypers.ADD_TOKEN_SECOND)

    model = PET(hypers, hypers.TRANSFORMER_D_MODEL, hypers.TRANSFORMER_N_HEAD,
                           hypers.TRANSFORMER_DIM_FEEDFORWARD, hypers.N_TRANS_LAYERS, 
                           0.0, len(all_species), 
                           hypers.N_GNN_LAYERS, hypers.HEAD_N_NEURONS, hypers.TRANSFORMERS_CENTRAL_SPECIFIC, hypers.HEADS_CENTRAL_SPECIFIC, 
                           add_tokens).to(device)

    if hypers.MULTI_GPU and torch.cuda.is_available():
        model = DataParallel(model)
        model = model.to( torch.device('cuda:0'))

    model.load_state_dict(torch.load(PATH_TO_MODEL_STATE_DICT))
    model.eval()

    if hypers.USE_ENERGIES:
        energies_ground_truth = np.array([struc.info[hypers.ENERGY_KEY] for struc in structures])

    if hypers.USE_FORCES:
        forces_ground_truth = [struc.arrays[hypers.FORCES_KEY] for struc in structures]
        forces_ground_truth = np.concatenate(forces_ground_truth, axis = 0)



    if hypers.USE_ENERGIES:
        all_energies_predicted = []

    if hypers.USE_FORCES:
        all_forces_predicted = []

    #warmup for correct time estimation
    for batch in loader:
        if not hypers.MULTI_GPU:
            batch.to(device)
            model.augmentation = USE_AUGMENTATION
        else:
            model.module.augmentation = USE_AUGMENTATION

        predictions_energies, targets_energies, predictions_forces, targets_forces = model(batch)
        break

    begin = time.time()
    for _ in tqdm(range(N_AUG)):
        if hypers.USE_ENERGIES:
            energies_predicted = []
        if hypers.USE_FORCES:
            forces_predicted = []

        for batch in loader:
            if not hypers.MULTI_GPU:
                batch.to(device)
                model.augmentation = USE_AUGMENTATION
            else:
                model.module.augmentation = USE_AUGMENTATION

            predictions_energies, targets_energies, predictions_forces, targets_forces = model(batch)
            if hypers.USE_ENERGIES:
                energies_predicted.append(predictions_energies.data.cpu().numpy())
            if hypers.USE_FORCES:
                forces_predicted.append(predictions_forces.data.cpu().numpy())

        if hypers.USE_ENERGIES:
            energies_predicted = np.concatenate(energies_predicted, axis = 0)
            all_energies_predicted.append(energies_predicted)

        if hypers.USE_FORCES:
            forces_predicted = np.concatenate(forces_predicted, axis = 0)
            all_forces_predicted.append(forces_predicted)

    total_time = time.time() - begin
    n_atoms = np.array([len(struc.positions) for struc in structures])
    time_per_atom = total_time / (np.sum(n_atoms) * N_AUG)

    if hypers.USE_ENERGIES:
        all_energies_predicted = [el[np.newaxis] for el in all_energies_predicted]
        all_energies_predicted = np.concatenate(all_energies_predicted, axis = 0)
        energies_predicted_mean = np.mean(all_energies_predicted, axis = 0)


        if all_energies_predicted.shape[0] > 1:
            energies_rotational_discrepancies = all_energies_predicted - energies_predicted_mean[np.newaxis]
            print('energies_rotational_discrepancies', energies_rotational_discrepancies.shape)
            energies_rotational_discrepancies_per_atom = energies_rotational_discrepancies / n_atoms[np.newaxis, :]
            correction = all_energies_predicted.shape[0] / (all_energies_predicted.shape[0] - 1)
            energies_rotational_std_per_atom = np.sqrt(np.mean(energies_rotational_discrepancies_per_atom ** 2) * correction)


        compositional_features = get_compositional_features(structures, all_species)
        self_contributions_energies = []
        for i in range(len(structures)):
            self_contributions_energies.append(np.dot(compositional_features[i], self_contributions))
        self_contributions_energies = np.array(self_contributions_energies)

        energies_predicted_mean = energies_predicted_mean + self_contributions_energies

        print(f"energies mae per struc: {get_mae(energies_ground_truth, energies_predicted_mean)}")
        print(f"energies rmse per struc: {get_rmse(energies_ground_truth, energies_predicted_mean)}")


        energies_predicted_mean_per_atom = energies_predicted_mean / n_atoms
        energies_ground_truth_per_atom = energies_ground_truth / n_atoms

        print(f"energies mae per atom: {get_mae(energies_ground_truth_per_atom, energies_predicted_mean_per_atom)}")
        print(f"energies rmse per atom: {get_rmse(energies_ground_truth_per_atom, energies_predicted_mean_per_atom)}")

        if all_energies_predicted.shape[0] > 1:
            if args.verbose:
                print(f"energies rotational discrepancy std per atom: {energies_rotational_std_per_atom}")


    if hypers.USE_FORCES:
        all_forces_predicted = [el[np.newaxis] for el in all_forces_predicted]
        all_forces_predicted = np.concatenate(all_forces_predicted, axis = 0)
        forces_predicted_mean = np.mean(all_forces_predicted, axis = 0)

        print(f"forces mae per component: {get_mae(forces_ground_truth, forces_predicted_mean)}")
        print(f"forces rmse per component: {get_rmse(forces_ground_truth, forces_predicted_mean)}")

        if all_forces_predicted.shape[0] > 1:
            forces_rotational_discrepancies = all_forces_predicted - forces_predicted_mean[np.newaxis]
            correction = all_forces_predicted.shape[0] / (all_forces_predicted.shape[0] - 1)
            forces_rotational_std = np.sqrt(np.mean(forces_rotational_discrepancies ** 2) * correction)
            if args.verbose:
                print(f"forces rotational discrepancy std per component: {forces_rotational_std} ")


    if args.verbose:
        print(f"approximate time per atom (batch size is {args.batch_size}): {time_per_atom} seconds")

    '''if hypers.USE_ENERGIES and not hypers.USE_FORCES:
        print(f"approximate time to compute energies per atom: {time_per_atom} seconds")
    else:
        print(f"approximate time to compute energies and forces per atom: {time_per_atom} seconds")'''


    if args.path_save_predictions is not None:
        if hypers.USE_ENERGIES:
            np.save(args.path_save_predictions + '/energies_predicted.npy', energies_predicted_mean)
        if hypers.USE_FORCES:
            np.save(args.path_save_predictions + '/forces_predicted.npy', forces_predicted_mean)



if __name__ == "__main__":
    main()
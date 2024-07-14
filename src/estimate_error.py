import torch
import ase.io
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader, DataListLoader
import time
from torch_geometric.nn import DataParallel
import argparse

from .hypers import load_hypers_from_file
from .pet import PET, PETMLIPWrapper, PETUtilityWrapper
from .utilities import (
    get_rmse,
    get_mae,
    set_reproducibility,
    Accumulator,
    report_accuracy,
)
from .data_preparation import get_pyg_graphs, get_compositional_features
from .data_preparation import get_targets
from .utilities import dtype2string, string2dtype
from .pet import FlagsWrapper


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "structures_path", help="Path to an xyz file with structures", type=str
    )
    parser.add_argument(
        "path_to_calc_folder", help="Path to a folder with a model to use", type=str
    )
    parser.add_argument(
        "checkpoint", help="Path to a particular checkpoint to use", type=str
    )

    parser.add_argument(
        "n_aug",
        type=int,
        help="A number of rotational augmentations to use. It should be a positive integer or -1. If -1, the initial coordinate system will be used, not a single random one, as in the n_aug = 1 case",
    )

    parser.add_argument(
        "batch_size",
        type=int,
        help="Batch size to use for inference. It should be a positive integer or -1. If -1, it will be set to the value used for fitting the provided model.",
    )

    parser.add_argument(
        "--path_save_predictions",
        help="Path to a folder where to save predictions.",
        type=str,
    )
    parser.add_argument("--verbose", help="Show more details", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        help="dtype to be used; one of 'float16', 'bfloat16', 'float32'.",
    )

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    HYPERS_PATH = args.path_to_calc_folder + "/hypers_used.yaml"
    PATH_TO_MODEL_STATE_DICT = (
        args.path_to_calc_folder + "/" + args.checkpoint + "_state_dict"
    )
    ALL_SPECIES_PATH = args.path_to_calc_folder + "/all_species.npy"
    SELF_CONTRIBUTIONS_PATH = args.path_to_calc_folder + "/self_contributions.npy"

    if args.n_aug == -1:
        N_AUG = 1
        USE_AUGMENTATION = False
    else:
        N_AUG = args.n_aug
        USE_AUGMENTATION = True

    hypers = load_hypers_from_file(HYPERS_PATH)
    if args.dtype is None:
        dtype = string2dtype(hypers.ARCHITECTURAL_HYPERS.DTYPE)
    else:
        dtype = string2dtype(args.dtype)
    torch.set_default_dtype(dtype)

    print(f"using {dtype2string(dtype)} for calculations")
    if hypers.UTILITY_FLAGS.CALCULATION_TYPE not in ["general_target", "mlip"]:
        raise ValueError("unknown calculation type")

    FITTING_SCHEME = hypers.FITTING_SCHEME

    ARCHITECTURAL_HYPERS = hypers.ARCHITECTURAL_HYPERS

    # set_reproducibility(FITTING_SCHEME.RANDOM_SEED, FITTING_SCHEME.CUDA_DETERMINISTIC)

    if args.batch_size == -1:
        args.batch_size = FITTING_SCHEME.STRUCTURAL_BATCH_SIZE

    structures = ase.io.read(args.structures_path, index=":")

    all_species = np.load(ALL_SPECIES_PATH)

    graphs = get_pyg_graphs(
        structures,
        all_species,
        ARCHITECTURAL_HYPERS.R_CUT,
        ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
        ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
        ARCHITECTURAL_HYPERS.K_CUT,
        ARCHITECTURAL_HYPERS.N_TARGETS > 1,
        ARCHITECTURAL_HYPERS.TARGET_INDEX_KEY
    )

    if FITTING_SCHEME.MULTI_GPU:
        loader = DataListLoader(graphs, batch_size=args.batch_size, shuffle=False)
    else:
        loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    model = PET(ARCHITECTURAL_HYPERS, 0.0, len(all_species)).to(device)
    model = PETUtilityWrapper(model, FITTING_SCHEME.GLOBAL_AUG)

    if hypers.UTILITY_FLAGS.CALCULATION_TYPE == "mlip":
        model = PETMLIPWrapper(
            model, hypers.MLIP_SETTINGS.USE_ENERGIES, hypers.MLIP_SETTINGS.USE_FORCES
        )

    if FITTING_SCHEME.MULTI_GPU and torch.cuda.is_available():
        model = DataParallel(FlagsWrapper(model))
        model = model.to(torch.device("cuda:0"))

    model.load_state_dict(torch.load(PATH_TO_MODEL_STATE_DICT))
    model = model.to(dtype=dtype)
    model.eval()

    # warmup for correct time estimation
    for batch in loader:
        if not FITTING_SCHEME.MULTI_GPU:
            batch.to(device)
        if hypers.UTILITY_FLAGS.CALCULATION_TYPE == "mlip":
            if FITTING_SCHEME.MULTI_GPU:
                model.module.augmentation = USE_AUGMENTATION
                model.module.create_graph = False
                _ = model(batch)
            else:
                _ = model(batch, augmentation=USE_AUGMENTATION, create_graph=False)
        else:
            _ = model(batch, augmentation=USE_AUGMENTATION)
        break

    begin = time.time()
    batch_accumulator = Accumulator()
    aug_accumulator = Accumulator()

    for _ in tqdm(range(N_AUG)):
        for batch in loader:
            if not FITTING_SCHEME.MULTI_GPU:
                batch.to(device)

            if hypers.UTILITY_FLAGS.CALCULATION_TYPE == "mlip":
                if FITTING_SCHEME.MULTI_GPU:
                    model.module.augmentation = USE_AUGMENTATION
                    model.module.create_graph = False
                    predictions_batch = model(batch)
                else:
                    predictions_batch = model(
                        batch, augmentation=USE_AUGMENTATION, create_graph=False
                    )
            else:
                predictions_batch = model(batch, augmentation=USE_AUGMENTATION)

            batch_accumulator.update(predictions_batch)
        predictions = batch_accumulator.flush()
        for index in range(len(predictions)):
            if predictions[index] is not None:
                predictions[index] = predictions[index][np.newaxis]
        aug_accumulator.update(predictions)

    all_predictions = aug_accumulator.flush()

    total_time = time.time() - begin
    n_atoms = np.array([len(struc.positions) for struc in structures])
    time_per_atom = total_time / (np.sum(n_atoms) * N_AUG)

    if hypers.UTILITY_FLAGS.CALCULATION_TYPE == "mlip":
        all_energies_predicted, all_forces_predicted = all_predictions
        MLIP_SETTINGS = hypers.MLIP_SETTINGS
        if MLIP_SETTINGS.USE_ENERGIES:
            self_contributions = np.load(SELF_CONTRIBUTIONS_PATH)
            energies_ground_truth = np.array(
                [struc.info[MLIP_SETTINGS.ENERGY_KEY] for struc in structures]
            )

            compositional_features = get_compositional_features(structures, all_species)
            self_contributions_energies = []
            for i in range(len(structures)):
                self_contributions_energies.append(
                    np.dot(compositional_features[i], self_contributions)
                )
            self_contributions_energies = np.array(self_contributions_energies)

            all_energies_predicted = (
                all_energies_predicted + self_contributions_energies[np.newaxis, :]
            )

            report_accuracy(
                all_energies_predicted,
                energies_ground_truth,
                "energies",
                args.verbose,
                specify_per_component=False,
                target_type="structural",
                n_atoms=n_atoms,
                support_missing_values=FITTING_SCHEME.SUPPORT_MISSING_VALUES,
            )

        if MLIP_SETTINGS.USE_FORCES:
            forces_ground_truth = [
                struc.arrays[MLIP_SETTINGS.FORCES_KEY] for struc in structures
            ]
            forces_ground_truth = np.concatenate(forces_ground_truth, axis=0)
            report_accuracy(
                all_forces_predicted,
                forces_ground_truth,
                "forces",
                args.verbose,
                specify_per_component=True,
                target_type="atomic",
                n_atoms=n_atoms,
                support_missing_values=FITTING_SCHEME.SUPPORT_MISSING_VALUES,
            )

        if args.path_save_predictions is not None:
            if MLIP_SETTINGS.USE_ENERGIES:
                energies_predicted_mean = np.mean(all_energies_predicted, axis=0)
                np.save(
                    args.path_save_predictions + "/energies_predicted.npy",
                    energies_predicted_mean,
                )
            if MLIP_SETTINGS.USE_FORCES:
                forces_predicted_mean = np.mean(all_forces_predicted, axis=0)
                np.save(
                    args.path_save_predictions + "/forces_predicted.npy",
                    forces_predicted_mean,
                )

    if hypers.UTILITY_FLAGS.CALCULATION_TYPE == "general_target":
        if len(all_predictions) != 1:
            raise ValueError("for general target model should predict only one target")
        all_targets_predicted = all_predictions[0]
        GENERAL_TARGET_SETTINGS = hypers.GENERAL_TARGET_SETTINGS
        ground_truth = get_targets(structures, GENERAL_TARGET_SETTINGS)
        ground_truth = [
            el.data.cpu().to(dtype=torch.float32).numpy() for el in ground_truth
        ]
        ground_truth = np.concatenate(ground_truth, axis=0)

        report_accuracy(
            all_targets_predicted,
            ground_truth,
            GENERAL_TARGET_SETTINGS.TARGET_KEY,
            args.verbose,
            specify_per_component=True,
            target_type=GENERAL_TARGET_SETTINGS.TARGET_TYPE,
            n_atoms=n_atoms,
            support_missing_values=FITTING_SCHEME.SUPPORT_MISSING_VALUES,
        )

        if args.path_save_predictions is not None:
            targets_predicted_mean = np.mean(all_targets_predicted, axis=0)
            np.save(
                args.path_save_predictions + "/targets_predicted.npy",
                targets_predicted_mean,
            )

    if args.verbose:
        print(
            f"approximate time per atom not including neighbor list construction for batch size of {args.batch_size}: {time_per_atom} seconds"
        )

    """if MLIP_SETTINGS.USE_ENERGIES and not MLIP_SETTINGS.USE_FORCES:
        print(f"approximate time to compute energies per atom: {time_per_atom} seconds")
    else:
        print(f"approximate time to compute energies and forces per atom: {time_per_atom} seconds")"""


if __name__ == "__main__":
    main()

from .data_preparation import get_all_species
import os

import torch
import ase.io
import numpy as np
from tqdm import tqdm
from .utilities import ModelKeeper
import time
import pickle
from torch_geometric.nn import DataParallel

from .hypers import save_hypers, set_hypers_from_files
from .pet import PET, PETUtilityWrapper
from .utilities import FullLogger, get_scheduler, load_checkpoint, get_data_loaders
from .utilities import get_loss, set_reproducibility, get_calc_names
from .utilities import get_optimizer
from .analysis import adapt_hypers
import argparse
from .data_preparation import get_pyg_graphs, update_pyg_graphs, get_targets
from .utilities import dtype2string, string2dtype


def main():
    TIME_SCRIPT_STARTED = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "train_structures_path",
        help="Path to an xyz file with train structures",
        type=str,
    )
    parser.add_argument(
        "val_structures_path",
        help="Path to an xyz file with validation structures",
        type=str,
    )
    parser.add_argument(
        "provided_hypers_path",
        help="Path to a YAML file with provided hypers",
        type=str,
    )
    parser.add_argument(
        "default_hypers_path", help="Path to a YAML file with default hypers", type=str
    )
    parser.add_argument(
        "name_of_calculation", help="Name of this calculation", type=str
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hypers = set_hypers_from_files(args.provided_hypers_path, args.default_hypers_path)
    dtype = string2dtype(hypers.ARCHITECTURAL_HYPERS.DTYPE)
    torch.set_default_dtype(dtype)

    FITTING_SCHEME = hypers.FITTING_SCHEME
    GENERAL_TARGET_SETTINGS = hypers.GENERAL_TARGET_SETTINGS
    ARCHITECTURAL_HYPERS = hypers.ARCHITECTURAL_HYPERS

    ARCHITECTURAL_HYPERS.D_OUTPUT = GENERAL_TARGET_SETTINGS.TARGET_DIM
    ARCHITECTURAL_HYPERS.TARGET_TYPE = GENERAL_TARGET_SETTINGS.TARGET_TYPE
    ARCHITECTURAL_HYPERS.TARGET_AGGREGATION = GENERAL_TARGET_SETTINGS.TARGET_AGGREGATION

    set_reproducibility(FITTING_SCHEME.RANDOM_SEED, FITTING_SCHEME.CUDA_DETERMINISTIC)

    train_structures = ase.io.read(args.train_structures_path, index=":")
    adapt_hypers(FITTING_SCHEME, train_structures)

    val_structures = ase.io.read(args.val_structures_path, index=":")
    structures = train_structures + val_structures
    all_species = get_all_species(structures)

    if "results" not in os.listdir("."):
        os.mkdir("results")

    name_to_load, NAME_OF_CALCULATION = get_calc_names(
        os.listdir("results"), args.name_of_calculation
    )

    os.mkdir(f"results/{NAME_OF_CALCULATION}")
    np.save(f"results/{NAME_OF_CALCULATION}/all_species.npy", all_species)
    hypers.UTILITY_FLAGS.CALCULATION_TYPE = "general_target"
    save_hypers(hypers, f"results/{NAME_OF_CALCULATION}/hypers_used.yaml")

    print(len(train_structures))
    print(len(val_structures))

    train_graphs = get_pyg_graphs(
        train_structures,
        all_species,
        ARCHITECTURAL_HYPERS.R_CUT,
        ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
        ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
        ARCHITECTURAL_HYPERS.K_CUT,
        ARCHITECTURAL_HYPERS.N_TARGETS > 1,
        ARCHITECTURAL_HYPERS.TARGET_INDEX_KEY
    )
    val_graphs = get_pyg_graphs(
        val_structures,
        all_species,
        ARCHITECTURAL_HYPERS.R_CUT,
        ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
        ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
        ARCHITECTURAL_HYPERS.K_CUT,
        ARCHITECTURAL_HYPERS.N_TARGETS > 1,
        ARCHITECTURAL_HYPERS.TARGET_INDEX_KEY
    )

    train_targets = get_targets(train_structures, GENERAL_TARGET_SETTINGS)
    val_targets = get_targets(val_structures, GENERAL_TARGET_SETTINGS)

    update_pyg_graphs(train_graphs, "targets", train_targets)
    update_pyg_graphs(val_graphs, "targets", val_targets)

    train_loader, val_loader = get_data_loaders(
        train_graphs, val_graphs, FITTING_SCHEME
    )

    model = PET(ARCHITECTURAL_HYPERS, 0.0, len(all_species)).to(device)
    model = PETUtilityWrapper(model, FITTING_SCHEME.GLOBAL_AUG)

    if FITTING_SCHEME.MULTI_GPU and torch.cuda.is_available():
        model = DataParallel(model)
        model = model.to(torch.device("cuda:0"))

    if FITTING_SCHEME.MODEL_TO_START_WITH is not None:
        model.load_state_dict(torch.load(FITTING_SCHEME.MODEL_TO_START_WITH))
        model = model.to(dtype=dtype)

    optim = get_optimizer(model, FITTING_SCHEME)
    scheduler = get_scheduler(optim, FITTING_SCHEME)

    if name_to_load is not None:
        load_checkpoint(model, optim, scheduler, f"results/{name_to_load}/checkpoint")

    history = []
    logger = FullLogger(FITTING_SCHEME.SUPPORT_MISSING_VALUES)
    mae_model_keeper = ModelKeeper()
    rmse_model_keeper = ModelKeeper()

    pbar = tqdm(range(FITTING_SCHEME.EPOCH_NUM))

    for epoch in pbar:

        model.train(True)
        for batch in train_loader:
            if not FITTING_SCHEME.MULTI_GPU:
                batch.to(device)

            predictions = model(batch, augmentation=True)
            logger.train_logger.update(predictions, batch.targets)
            loss = get_loss(
                predictions,
                batch.targets,
                FITTING_SCHEME.SUPPORT_MISSING_VALUES,
                FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
            )
            loss.backward()
            if FITTING_SCHEME.DO_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=FITTING_SCHEME.GRADIENT_CLIPPING_MAX_NORM,
                )
            optim.step()
            optim.zero_grad()

        model.train(False)
        for batch in val_loader:
            if not FITTING_SCHEME.MULTI_GPU:
                batch.to(device)

            predictions = model(batch, augmentation=False)
            logger.val_logger.update(predictions, batch.targets)

        now = {}
        now["errors"] = logger.flush()

        now["lr"] = scheduler.get_last_lr()
        now["epoch"] = epoch
        now["elapsed_time"] = time.time() - TIME_SCRIPT_STARTED

        mae_model_keeper.update(model, now["errors"]["val"]["mae"], epoch)
        rmse_model_keeper.update(model, now["errors"]["val"]["rmse"], epoch)

        val_mae_message = "val mae/rmse:"
        train_mae_message = "train mae/rmse:"

        val_mae_message += (
            f" {now['errors']['val']['mae']}/{now['errors']['val']['rmse']};"
        )
        train_mae_message += (
            f" {now['errors']['train']['mae']}/{now['errors']['train']['rmse']};"
        )

        pbar.set_description(
            f"lr: {scheduler.get_last_lr()}; " + val_mae_message + train_mae_message
        )

        history.append(now)
        scheduler.step()
        elapsed = time.time() - TIME_SCRIPT_STARTED
        if FITTING_SCHEME.MAX_TIME is not None:
            if elapsed > FITTING_SCHEME.MAX_TIME:
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "dtype_used": dtype2string(dtype),
        },
        f"results/{NAME_OF_CALCULATION}/checkpoint",
    )
    with open(f"results/{NAME_OF_CALCULATION}/history.pickle", "wb") as f:
        pickle.dump(history, f)

    def save_model(model_name, model_keeper):
        torch.save(
            model_keeper.best_model.state_dict(),
            f"results/{NAME_OF_CALCULATION}/{model_name}_state_dict",
        )

    summary = ""
    save_model("best_val_mae_model", mae_model_keeper)
    summary += f"best val mae: {mae_model_keeper.best_error} at epoch {mae_model_keeper.best_epoch}\n"

    save_model("best_val_rmse_model", rmse_model_keeper)
    summary += f"best val rmse: {rmse_model_keeper.best_error} at epoch {rmse_model_keeper.best_epoch}\n"

    with open(f"results/{NAME_OF_CALCULATION}/summary.txt", "w") as f:
        print(summary, file=f)

    print("total elapsed time: ", time.time() - TIME_SCRIPT_STARTED)


if __name__ == "__main__":
    main()

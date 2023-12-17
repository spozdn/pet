from .data_preparation import get_all_species
import os

import torch
import ase.io
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader, DataListLoader
from .utilities import ModelKeeper
import time
from torch.optim.lr_scheduler import LambdaLR
import random
from torch_geometric.nn import DataParallel

from .hypers import Hypers, save_hypers
from .pet import PET
from .utilities import FullLogger
from .utilities import get_rmse, get_loss, set_reproducibility, get_calc_names
from .analysis import adapt_hypers
from .data_preparation import get_self_contributions, get_corrected_energies
import argparse
from .data_preparation import get_pyg_graphs, update_pyg_graphs, get_forces

def main():
    TIME_SCRIPT_STARTED = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("train_structures_path", help="Path to an xyz file with train structures", type = str)
    parser.add_argument("val_structures_path", help="Path to an xyz file with validation structures", type = str)
    parser.add_argument("provided_hypers_path", help="Path to a YAML file with provided hypers", type = str)
    parser.add_argument("default_hypers_path", help="Path to a YAML file with default hypers", type = str)
    parser.add_argument("name_of_calculation", help="Name of this calculation", type = str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hypers = Hypers()
    hypers.set_from_files(args.provided_hypers_path, args.default_hypers_path)
    set_reproducibility(hypers.RANDOM_SEED, hypers.CUDA_DETERMINISTIC)

    train_structures = ase.io.read(args.train_structures_path, index = ':')
    adapt_hypers(hypers, train_structures)

    val_structures = ase.io.read(args.val_structures_path, index = ':')
    structures = train_structures + val_structures 
    all_species = get_all_species(structures)

    if 'results' not in os.listdir('.'):
        os.mkdir('results')
    
    name_to_load, NAME_OF_CALCULATION = get_calc_names(os.listdir('results'), args.name_of_calculation)

    os.mkdir(f'results/{NAME_OF_CALCULATION}')
    np.save(f'results/{NAME_OF_CALCULATION}/all_species.npy', all_species)
    save_hypers(hypers, f"results/{NAME_OF_CALCULATION}/hypers_used.yaml")

    print(len(train_structures))
    print(len(val_structures))

    train_graphs = get_pyg_graphs(train_structures, all_species, hypers.R_CUT, hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES)
    val_graphs = get_pyg_graphs(val_structures, all_species, hypers.R_CUT, hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES)

    if hypers.USE_ENERGIES:
        self_contributions = get_self_contributions(hypers.ENERGY_KEY, train_structures, all_species)
        np.save(f'results/{NAME_OF_CALCULATION}/self_contributions.npy', self_contributions)

        train_energies = get_corrected_energies(hypers.ENERGY_KEY, train_structures, all_species, self_contributions)
        val_energies = get_corrected_energies(hypers.ENERGY_KEY, val_structures, all_species, self_contributions)

        update_pyg_graphs(train_graphs, 'y', train_energies)
        update_pyg_graphs(val_graphs, 'y', val_energies)

    if hypers.USE_FORCES:
        train_forces = get_forces(train_structures, hypers.FORCES_KEY)
        val_forces = get_forces(val_structures, hypers.FORCES_KEY)

        update_pyg_graphs(train_graphs, 'forces', train_forces)
        update_pyg_graphs(val_graphs, 'forces', val_forces)


    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(hypers.RANDOM_SEED)

    if hypers.MULTI_GPU:
        train_loader = DataListLoader(train_graphs, batch_size=hypers.STRUCTURAL_BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
        val_loader = DataListLoader(val_graphs, batch_size = hypers.STRUCTURAL_BATCH_SIZE, shuffle = False, worker_init_fn=seed_worker, generator=g)
    else:
        train_loader = DataLoader(train_graphs, batch_size=hypers.STRUCTURAL_BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_graphs, batch_size = hypers.STRUCTURAL_BATCH_SIZE, shuffle = False, worker_init_fn=seed_worker, generator=g)


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
        model = model.to(torch.device('cuda:0'))
    
    optim = torch.optim.Adam(model.parameters(), lr = hypers.INITIAL_LR)

    def func_lr_scheduler(epoch):
        if epoch < hypers.EPOCHS_WARMUP:
            return epoch / hypers.EPOCHS_WARMUP
        delta = epoch - hypers.EPOCHS_WARMUP
        num_blocks = delta // hypers.SCHEDULER_STEP_SIZE 
        return 0.5 ** (num_blocks)

    scheduler = LambdaLR(optim, func_lr_scheduler)


    if hypers.MODEL_TO_START_WITH is not None:
        model.load_state_dict(torch.load(hypers.MODEL_TO_START_WITH))

    if name_to_load is not None:
        checkpoint = torch.load(f'results/{name_to_load}/checkpoint')

        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optim_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    history = []
    if hypers.USE_ENERGIES:
        energies_logger = FullLogger()

    if hypers.USE_FORCES:
        forces_logger = FullLogger()



    if hypers.USE_FORCES:
        all_val_forces = []
        model.train(False)
        for batch in val_loader:
            if not hypers.MULTI_GPU:
                batch.to(device)
                model.augmentation = False
            else:
                model.module.augmentation = False

            _, _, _, targets_forces = model(batch)
            all_val_forces.append(targets_forces.data.cpu().numpy())
        all_val_forces = np.concatenate(all_val_forces, axis = 0)

        sliding_forces_rmse = get_rmse(all_val_forces, 0.0)

        forces_rmse_model_keeper = ModelKeeper()
        forces_mae_model_keeper = ModelKeeper()

    if hypers.USE_ENERGIES:
        sliding_energies_rmse = get_rmse(val_energies, np.mean(val_energies))

        energies_rmse_model_keeper = ModelKeeper()
        energies_mae_model_keeper = ModelKeeper()

    if hypers.USE_ENERGIES and hypers.USE_FORCES:
        multiplication_rmse_model_keeper = ModelKeeper()
        multiplication_mae_model_keeper = ModelKeeper()

    best_val_mae = None
    best_val_model = None
    pbar = tqdm(range(hypers.EPOCH_NUM))



    for epoch in pbar:

        model.train(True)
        for batch in train_loader:
            if not hypers.MULTI_GPU:
                batch.to(device)
                model.augmentation = True
            else:
                model.module.augmentation = True

            predictions_energies, targets_energies, predictions_forces, targets_forces = model(batch)
            if hypers.USE_ENERGIES:
                energies_logger.train_logger.update(predictions_energies, targets_energies)
                loss_energies = get_loss(predictions_energies, targets_energies)
            if hypers.USE_FORCES:
                forces_logger.train_logger.update(predictions_forces, targets_forces)
                loss_forces = get_loss(predictions_forces, targets_forces)

            if hypers.USE_ENERGIES and hypers.USE_FORCES: 
                loss = hypers.ENERGY_WEIGHT * loss_energies / (sliding_energies_rmse ** 2) + loss_forces / (sliding_forces_rmse ** 2)
                loss.backward()

            if hypers.USE_ENERGIES and (not hypers.USE_FORCES):
                loss_energies.backward()
            if hypers.USE_FORCES and (not hypers.USE_ENERGIES):
                loss_forces.backward()


            optim.step()
            optim.zero_grad()

        model.train(False)
        for batch in val_loader:
            if not hypers.MULTI_GPU:
                batch.to(device)
                model.augmentation = False
            else:
                model.module.augmentation = False

            predictions_energies, targets_energies, predictions_forces, targets_forces = model(batch)
            if hypers.USE_ENERGIES:
                energies_logger.val_logger.update(predictions_energies, targets_energies)
            if hypers.USE_FORCES:
                forces_logger.val_logger.update(predictions_forces, targets_forces)

        now = {}
        if hypers.USE_ENERGIES:
            now['energies'] = energies_logger.flush()
        if hypers.USE_FORCES:
            now['forces'] = forces_logger.flush()   
        now['lr'] = scheduler.get_last_lr()
        now['epoch'] = epoch
        now['elapsed_time'] = time.time() - TIME_SCRIPT_STARTED

        if hypers.USE_ENERGIES:
            sliding_energies_rmse = hypers.SLIDING_FACTOR * sliding_energies_rmse + (1.0 - hypers.SLIDING_FACTOR) * now['energies']['val']['rmse']

            energies_mae_model_keeper.update(model, now['energies']['val']['mae'], epoch)
            energies_rmse_model_keeper.update(model, now['energies']['val']['rmse'], epoch)


        if hypers.USE_FORCES:
            sliding_forces_rmse = hypers.SLIDING_FACTOR * sliding_forces_rmse + (1.0 - hypers.SLIDING_FACTOR) * now['forces']['val']['rmse']
            forces_mae_model_keeper.update(model, now['forces']['val']['mae'], epoch)
            forces_rmse_model_keeper.update(model, now['forces']['val']['rmse'], epoch)    

        if hypers.USE_ENERGIES and hypers.USE_FORCES:
            multiplication_mae_model_keeper.update(model, now['forces']['val']['mae'] * now['energies']['val']['mae'], epoch,
                                                   additional_info = [now['energies']['val']['mae'], now['forces']['val']['mae']])
            multiplication_rmse_model_keeper.update(model, now['forces']['val']['rmse'] * now['energies']['val']['rmse'], epoch,
                                                    additional_info = [now['energies']['val']['rmse'], now['forces']['val']['rmse']])


        val_mae_message = "val mae/rmse:"
        train_mae_message = "train mae/rmse:"

        if hypers.USE_ENERGIES:
            val_mae_message += f" {now['energies']['val']['mae']}/{now['energies']['val']['rmse']};"
            train_mae_message += f" {now['energies']['train']['mae']}/{now['energies']['train']['rmse']};"
        if hypers.USE_FORCES:
            val_mae_message += f" {now['forces']['val']['mae']}/{now['forces']['val']['rmse']}"
            train_mae_message += f" {now['forces']['train']['mae']}/{now['forces']['train']['rmse']}"

        pbar.set_description(f"lr: {scheduler.get_last_lr()}; " + val_mae_message + train_mae_message)

        history.append(now)
        scheduler.step()
        elapsed = time.time() - TIME_SCRIPT_STARTED
        if hypers.MAX_TIME is not None:
            if elapsed > hypers.MAX_TIME:
                break

    import pickle
    torch.save({
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                }, f'results/{NAME_OF_CALCULATION}/checkpoint')
    with open(f'results/{NAME_OF_CALCULATION}/history.pickle', 'wb') as f:
        pickle.dump(history, f)

    torch.save(model.state_dict(), f'results/{NAME_OF_CALCULATION}/last_model_state_dict')
    # torch.save(model, f'results/{NAME_OF_CALCULATION}/last_model')

    def save_model(model_name, model_keeper):
        torch.save(model_keeper.best_model.state_dict(), f'results/{NAME_OF_CALCULATION}/{model_name}_state_dict')
        # torch.save(model_keeper.best_model, f'results/{NAME_OF_CALCULATION}/{model_name}')

    summary = ''
    if hypers.USE_ENERGIES:    
        save_model('best_val_mae_energies_model', energies_mae_model_keeper)
        summary += f'best val mae in energies: {energies_mae_model_keeper.best_error} at epoch {energies_mae_model_keeper.best_epoch}\n'

        save_model('best_val_rmse_energies_model', energies_rmse_model_keeper)
        summary += f'best val rmse in energies: {energies_rmse_model_keeper.best_error} at epoch {energies_rmse_model_keeper.best_epoch}\n'

    if hypers.USE_FORCES:
        save_model('best_val_mae_forces_model', forces_mae_model_keeper)
        summary += f'best val mae in forces: {forces_mae_model_keeper.best_error} at epoch {forces_mae_model_keeper.best_epoch}\n'

        save_model('best_val_rmse_forces_model', forces_rmse_model_keeper)
        summary += f'best val rmse in forces: {forces_rmse_model_keeper.best_error} at epoch {forces_rmse_model_keeper.best_epoch}\n'

    if hypers.USE_ENERGIES and hypers.USE_FORCES:
        save_model('best_val_mae_both_model', multiplication_mae_model_keeper)
        summary += f'best both (multiplication) mae in energies: {multiplication_mae_model_keeper.additional_info[0]} in forces: {multiplication_mae_model_keeper.additional_info[1]} at epoch {multiplication_mae_model_keeper.best_epoch}\n'


        save_model('best_val_rmse_both_model', multiplication_rmse_model_keeper)
        summary += f'best both (multiplication) rmse in energies: {multiplication_rmse_model_keeper.additional_info[0]} in forces: {multiplication_rmse_model_keeper.additional_info[1]} at epoch {multiplication_rmse_model_keeper.best_epoch}\n'

    with open(f"results/{NAME_OF_CALCULATION}/summary.txt", 'w') as f:
        print(summary, file = f)
    
    print("total elapsed time: ", time.time() - TIME_SCRIPT_STARTED)

if __name__ == "__main__":
    main()
    

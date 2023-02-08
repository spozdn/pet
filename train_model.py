from utilities import get_all_species, get_compositional_features
import os

import torch
import ase.io
import numpy as np
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from utilities import ModelKeeper
import time
from scipy.spatial.transform import Rotation
from torch.optim.lr_scheduler import LambdaLR
import sys
import copy
import inspect
import yaml
import random

from molecule import Molecule, batch_to_dict
from hypers import Hypers
from pet import PET
from utilities import FullLogger
from utilities import get_rmse, get_mae, get_relative_rmse, get_loss
from analysis import get_structural_batch_size, convert_atomic_throughput

TIME_SCRIPT_STARTED = time.time()

TRAIN_STRUCTURES_PATH = sys.argv[1]
VAL_STRUCTURES_PATH = sys.argv[2]

PROVIDED_HYPERS_PATH = sys.argv[3]
DEFAULT_HYPERS_PATH = sys.argv[4]
NAME_OF_CALCULATION = sys.argv[5]

Hypers.set_from_files(PROVIDED_HYPERS_PATH, DEFAULT_HYPERS_PATH)

#TRAIN_STRUCTURES = '../experiments/hme21_iteration_3/hme21_train.xyz'
#VAL_STRUCTURES = '../experiments/hme21_iteration_3/hme21_val.xyz'

torch.manual_seed(Hypers.RANDOM_SEED)
np.random.seed(Hypers.RANDOM_SEED)
random.seed(Hypers.RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(Hypers.RANDOM_SEED)
torch.cuda.manual_seed(Hypers.RANDOM_SEED)
torch.cuda.manual_seed_all(Hypers.RANDOM_SEED)

if Hypers.CUDA_DETERMINISTIC:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    
train_structures = ase.io.read(TRAIN_STRUCTURES_PATH, index = ':')

    
if 'STRUCTURAL_BATCH_SIZE' not in Hypers.__dict__.keys():
    Hypers.STRUCTURAL_BATCH_SIZE = get_structural_batch_size(train_structures, Hypers.ATOMIC_BATCH_SIZE)
    
if 'EPOCH_NUM' not in Hypers.__dict__.keys():
    Hypers.EPOCH_NUM = convert_atomic_throughput(train_structures, Hypers.EPOCH_NUM_ATOMIC)
    
if 'SCHEDULER_STEP_SIZE' not in Hypers.__dict__.keys():
    Hypers.SCHEDULER_STEP_SIZE = convert_atomic_throughput(train_structures, Hypers.SCHEDULER_STEP_SIZE_ATOMIC)
    
if 'EPOCHS_WARMUP' not in Hypers.__dict__.keys():
    Hypers.EPOCHS_WARMUP = convert_atomic_throughput(train_structures, Hypers.EPOCHS_WARMUP_ATOMIC)
    
    
val_structures = ase.io.read(VAL_STRUCTURES_PATH, index = ':')
structures = train_structures + val_structures 
all_species = get_all_species(structures)

results = os.listdir('results')
name_to_load = None

if NAME_OF_CALCULATION in results:
    name_to_load = NAME_OF_CALCULATION
    for i in range(100000):
        name_now = NAME_OF_CALCULATION + f'_continuation_{i}'
        if name_now not in results:
            name_to_save = name_now
            break
        name_to_load = name_now   
    NAME_OF_CALCULATION = name_to_save

    
    
os.mkdir(f'results/{NAME_OF_CALCULATION}')

np.save(f'results/{NAME_OF_CALCULATION}/all_species.npy', all_species)

all_members = inspect.getmembers(Hypers, lambda member:not(inspect.isroutine(member)))
all_hypers = []
for member in all_members:
    if member[0].startswith('__'):
        continue
    if member[0] == 'is_set':
        continue
    all_hypers.append(member)
all_hypers = {hyper[0] : hyper[1] for hyper in all_hypers}

with open(f"results/{NAME_OF_CALCULATION}/hypers_used.yaml", "w") as f:
    yaml.dump(all_hypers, f)
    
print(len(train_structures))
print(len(val_structures))

if Hypers.USE_ENERGIES:
    train_energies = np.array([structure.info['energy'] for structure in train_structures])
    val_energies = np.array([structure.info['energy'] for structure in val_structures])

    train_c_feat = get_compositional_features(train_structures, all_species)
    val_c_feat = get_compositional_features(val_structures, all_species)
    print(train_c_feat.shape)

    print(np.mean(np.abs(val_energies)))
    rgr = Ridge(alpha = 1e-10, fit_intercept = False)
    rgr.fit(train_c_feat, train_energies)
    train_energies -= rgr.predict(train_c_feat)
    val_energies -= rgr.predict(val_c_feat)
    print(np.mean(np.abs(val_energies)))
    np.save(f'results/{NAME_OF_CALCULATION}/self_contributions.npy', rgr.coef_)

train_molecules = [Molecule(structure, Hypers.R_CUT) for structure in tqdm(train_structures)]
val_molecules = [Molecule(structure, Hypers.R_CUT) for structure in tqdm(val_structures)]


molecules = train_molecules + val_molecules
max_nums = [molecule.get_max_num() for molecule in molecules]
max_num = np.max(max_nums)
print(max_num)

central_species = [molecule.central_species for molecule in molecules]
central_species = np.concatenate(central_species, axis = 0)

train_graphs = [molecule.get_graph(max_num, all_species) for molecule in tqdm(train_molecules)]
val_graphs = [molecule.get_graph(max_num, all_species) for molecule in tqdm(val_molecules)]


if Hypers.USE_ENERGIES:
    for index in range(len(train_structures)):
        train_graphs[index].y = train_energies[index]

    for index in range(len(val_structures)):
        val_graphs[index].y = val_energies[index]
    

    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(Hypers.RANDOM_SEED)

if Hypers.MULTI_GPU:
    train_loader = DataListLoader(train_graphs, batch_size=Hypers.STRUCTURAL_BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataListLoader(val_graphs, batch_size = Hypers.STRUCTURAL_BATCH_SIZE, shuffle = False, worker_init_fn=seed_worker, generator=g)
else:
    train_loader = DataLoader(train_graphs, batch_size=Hypers.STRUCTURAL_BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_graphs, batch_size = Hypers.STRUCTURAL_BATCH_SIZE, shuffle = False, worker_init_fn=seed_worker, generator=g)


add_tokens = []
for _ in range(Hypers.N_GNN_LAYERS - 1):
    add_tokens.append(Hypers.ADD_TOKEN_FIRST)
add_tokens.append(Hypers.ADD_TOKEN_SECOND)

model = PET(Hypers.TRANSFORMER_D_MODEL, Hypers.TRANSFORMER_N_HEAD,
                       Hypers.TRANSFORMER_DIM_FEEDFORWARD, Hypers.N_TRANS_LAYERS, 
                       0.0, len(all_species), 
                       Hypers.N_GNN_LAYERS, Hypers.HEAD_N_NEURONS, Hypers.TRANSFORMERS_CENTRAL_SPECIFIC, Hypers.HEADS_CENTRAL_SPECIFIC, 
                       add_tokens).cuda()

if Hypers.MULTI_GPU:
    model = DataParallel(model)
    device = torch.device('cuda:0')
    model = model.to(device)


import copy
optim = torch.optim.Adam(model.parameters(), lr = Hypers.INITIAL_LR)

def func_lr_scheduler(epoch):
    if epoch < Hypers.EPOCHS_WARMUP:
        return epoch / Hypers.EPOCHS_WARMUP
    delta = epoch - Hypers.EPOCHS_WARMUP
    num_blocks = delta // Hypers.SCHEDULER_STEP_SIZE 
    return 0.5 ** (num_blocks)

scheduler = LambdaLR(optim, func_lr_scheduler)

if name_to_load is not None:
    checkpoint = torch.load(f'results/{name_to_load}/checkpoint')

    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optim_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    
history = []
if Hypers.USE_ENERGIES:
    energies_logger = FullLogger()
    
if Hypers.USE_FORCES:
    forces_logger = FullLogger()



if Hypers.USE_FORCES:
    all_val_forces = []
    model.train(False)
    for batch in val_loader:
        batch.cuda()
        model.augmentation = False
        _, _, _, targets_forces = model(batch)
        all_val_forces.append(targets_forces.data.cpu().numpy())
    all_val_forces = np.concatenate(all_val_forces, axis = 0)

    sliding_forces_rmse = get_rmse(all_val_forces, 0.0)
    
    forces_rmse_model_keeper = ModelKeeper()
    forces_mae_model_keeper = ModelKeeper()

if Hypers.USE_ENERGIES:
    sliding_energies_rmse = get_rmse(val_energies, np.mean(val_energies))
    
    energies_rmse_model_keeper = ModelKeeper()
    energies_mae_model_keeper = ModelKeeper()

if Hypers.USE_ENERGIES and Hypers.USE_FORCES:
    multiplication_rmse_model_keeper = ModelKeeper()
    multiplication_mae_model_keeper = ModelKeeper()
    
best_val_mae = None
best_val_model = None
pbar = tqdm(range(Hypers.EPOCH_NUM))


    
for epoch in pbar:

    model.train(True)
    for batch in train_loader:
        batch.cuda()
        model.augmentation = True
        predictions_energies, targets_energies, predictions_forces, targets_forces = model(batch)
        if Hypers.USE_ENERGIES:
            energies_logger.train_logger.update(predictions_energies, targets_energies)
            loss_energies = get_loss(predictions_energies, targets_energies)
        if Hypers.USE_FORCES:
            forces_logger.train_logger.update(predictions_forces, targets_forces)
            loss_forces = get_loss(predictions_forces, targets_forces)

        if Hypers.USE_ENERGIES and Hypers.USE_FORCES: 
            loss = Hypers.ENERGY_WEIGHT * loss_energies / (sliding_energies_rmse ** 2) + loss_forces / (sliding_forces_rmse ** 2)
            loss.backward()

        if Hypers.USE_ENERGIES and (not Hypers.USE_FORCES):
            loss_energies.backward()
        if Hypers.USE_FORCES and (not Hypers.USE_ENERGIES):
            loss_forces.backward()


        optim.step()
        optim.zero_grad()

    model.train(False)
    for batch in val_loader:
        batch.cuda()
        model.augmentation = False
        predictions_energies, targets_energies, predictions_forces, targets_forces = model(batch)
        if Hypers.USE_ENERGIES:
            energies_logger.val_logger.update(predictions_energies, targets_energies)
        if Hypers.USE_FORCES:
            forces_logger.val_logger.update(predictions_forces, targets_forces)

    now = {}
    if Hypers.USE_ENERGIES:
        now['energies'] = energies_logger.flush()
    if Hypers.USE_FORCES:
        now['forces'] = forces_logger.flush()   
    now['lr'] = scheduler.get_last_lr()
    now['epoch'] = epoch

    if Hypers.USE_ENERGIES:
        sliding_energies_rmse = Hypers.SLIDING_FACTOR * sliding_energies_rmse + (1.0 - Hypers.SLIDING_FACTOR) * now['energies']['val']['rmse']

        energies_mae_model_keeper.update(model, now['energies']['val']['mae'], epoch)
        energies_rmse_model_keeper.update(model, now['energies']['val']['rmse'], epoch)


    if Hypers.USE_FORCES:
        sliding_forces_rmse = Hypers.SLIDING_FACTOR * sliding_forces_rmse + (1.0 - Hypers.SLIDING_FACTOR) * now['forces']['val']['rmse']
        forces_mae_model_keeper.update(model, now['forces']['val']['mae'], epoch)
        forces_rmse_model_keeper.update(model, now['forces']['val']['rmse'], epoch)    

    if Hypers.USE_ENERGIES and Hypers.USE_FORCES:
        multiplication_mae_model_keeper.update(model, now['forces']['val']['mae'] * now['energies']['val']['mae'], epoch,
                                               additional_info = [now['energies']['val']['mae'], now['forces']['val']['mae']])
        multiplication_rmse_model_keeper.update(model, now['forces']['val']['rmse'] * now['energies']['val']['rmse'], epoch,
                                                additional_info = [now['energies']['val']['rmse'], now['forces']['val']['rmse']])


    val_mae_message = "val mae/rmse:"
    train_mae_message = "train mae/rmse:"

    if Hypers.USE_ENERGIES:
        val_mae_message += f" {now['energies']['val']['mae']}/{now['energies']['val']['rmse']};"
        train_mae_message += f" {now['energies']['train']['mae']}/{now['energies']['train']['rmse']};"
    if Hypers.USE_FORCES:
        val_mae_message += f" {now['forces']['val']['mae']}/{now['forces']['val']['rmse']}"
        train_mae_message += f" {now['forces']['train']['mae']}/{now['forces']['train']['rmse']}"

    pbar.set_description(f"lr: {scheduler.get_last_lr()}; " + val_mae_message + train_mae_message)

    history.append(now)
    scheduler.step()
    elapsed = time.time() - TIME_SCRIPT_STARTED
    if Hypers.MAX_TIME is not None:
        if elapsed > Hypers.MAX_TIME:
            break
        
import os
import pickle
torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            }, f'results/{NAME_OF_CALCULATION}/checkpoint')
with open(f'results/{NAME_OF_CALCULATION}/history.pickle', 'wb') as f:
    pickle.dump(history, f)
    
torch.save(model.state_dict(), f'results/{NAME_OF_CALCULATION}/last_model_state_dict')
torch.save(model, f'results/{NAME_OF_CALCULATION}/last_model')

def save_model(model_name, model_keeper):
    torch.save(model_keeper.best_model.state_dict(), f'results/{NAME_OF_CALCULATION}/{model_name}_state_dict')
    torch.save(model_keeper.best_model, f'results/{NAME_OF_CALCULATION}/{model_name}')

summary = ''
if Hypers.USE_ENERGIES:    
    save_model('best_val_mae_energies_model', energies_mae_model_keeper)
    summary += f'best val mae in energies: {energies_mae_model_keeper.best_error} at epoch {energies_mae_model_keeper.best_epoch}\n'
    
    save_model('best_val_rmse_energies_model', energies_rmse_model_keeper)
    summary += f'best val rmse in energies: {energies_rmse_model_keeper.best_error} at epoch {energies_rmse_model_keeper.best_epoch}\n'
    
if Hypers.USE_FORCES:
    save_model('best_val_mae_forces_model', forces_mae_model_keeper)
    summary += f'best val mae in forces: {forces_mae_model_keeper.best_error} at epoch {forces_mae_model_keeper.best_epoch}\n'
    
    save_model('best_val_rmse_forces_model', forces_rmse_model_keeper)
    summary += f'best val rmse in forces: {forces_rmse_model_keeper.best_error} at epoch {forces_rmse_model_keeper.best_epoch}\n'
    
if Hypers.USE_ENERGIES and Hypers.USE_FORCES:
    save_model('best_val_mae_multiplication_model', multiplication_mae_model_keeper)
    summary += f'for best multiplication mae in energies: {multiplication_mae_model_keeper.additional_info[0]} in forces: {multiplication_mae_model_keeper.additional_info[1]} at epoch {multiplication_mae_model_keeper.best_epoch}\n'
    
    
    save_model('best_val_rmse_multiplication_model', multiplication_rmse_model_keeper)
    summary += f'for best multiplication rmse in energies: {multiplication_rmse_model_keeper.additional_info[0]} in forces: {multiplication_rmse_model_keeper.additional_info[1]} at epoch {multiplication_rmse_model_keeper.best_epoch}\n'
    
with open(f"results/{NAME_OF_CALCULATION}/summary.txt", 'w') as f:
    print(summary, file = f)
    


    

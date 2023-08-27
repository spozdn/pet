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
from torch_geometric.nn import DataParallel

from molecule import Molecule, batch_to_dict
from hypers import Hypers
from pet import PET
from utilities import FullLogger
from utilities import get_rmse, get_mae, get_relative_rmse, get_loss
from analysis import get_structural_batch_size, convert_atomic_throughput
import argparse
import math



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

#TRAIN_STRUCTURES = '../experiments/hme21_iteration_3/hme21_train.xyz'
#VAL_STRUCTURES = '../experiments/hme21_iteration_3/hme21_val.xyz'

torch.manual_seed(hypers.RANDOM_SEED)
np.random.seed(hypers.RANDOM_SEED)
random.seed(hypers.RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(hypers.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(hypers.RANDOM_SEED)
    torch.cuda.manual_seed_all(hypers.RANDOM_SEED)

if hypers.CUDA_DETERMINISTIC and torch.cuda.is_available():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
train_structures = ase.io.read(args.train_structures_path, index = ':')

    
if 'STRUCTURAL_BATCH_SIZE' not in hypers.__dict__.keys():
    hypers.STRUCTURAL_BATCH_SIZE = get_structural_batch_size(train_structures, hypers.ATOMIC_BATCH_SIZE)
    
if 'EPOCH_NUM' not in hypers.__dict__.keys():
    hypers.EPOCH_NUM = convert_atomic_throughput(train_structures, hypers.EPOCH_NUM_ATOMIC)
    
if 'SCHEDULER_STEP_SIZE' not in hypers.__dict__.keys():
    hypers.SCHEDULER_STEP_SIZE = convert_atomic_throughput(train_structures, hypers.SCHEDULER_STEP_SIZE_ATOMIC)
    
if 'EPOCHS_WARMUP' not in hypers.__dict__.keys():
    hypers.EPOCHS_WARMUP = convert_atomic_throughput(train_structures, hypers.EPOCHS_WARMUP_ATOMIC)
    
    
val_structures = ase.io.read(args.val_structures_path, index = ':')
structures = train_structures + val_structures 
all_species = get_all_species(structures)

if 'results' not in os.listdir('.'):
    os.mkdir('results')
results = os.listdir('results')
name_to_load = None
NAME_OF_CALCULATION = args.name_of_calculation
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

all_members = inspect.getmembers(hypers, lambda member:not(inspect.isroutine(member)))
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

def fit_with_nans(X, Y, alpha = 1e-10):
    rgr = Ridge(fit_intercept = False, alpha = alpha)
    rgr.fit(X, np.zeros_like(Y))
    
    for target_index in range(Y.shape[1]):
        y_now = Y[:, target_index]
        mask_now = np.logical_not(np.isnan(y_now))
        X_now = X[mask_now]
        y_now_masked = y_now[mask_now]
        if len(y_now_masked) == 0:
            raise ValueError(f"No data at all for the index {target_index}")
            
        rgr_tmp = Ridge(fit_intercept = False, alpha = alpha)
        rgr_tmp.fit(X_now, y_now_masked)
        rgr.coef_[target_index] = rgr_tmp.coef_
        
    return rgr

if hypers.USE_DIRECT_TARGETS:
    train_direct_targets = np.array([structure.info[hypers.TARGET_NAME] for structure in train_structures])
    val_direct_targets = np.array([structure.info[hypers.TARGET_NAME] for structure in val_structures])
    if len(train_direct_targets.shape) == 1:
        train_direct_targets = train_direct_targets[:, np.newaxis]
    if len(val_direct_targets.shape) == 1:
        val_direct_targets = val_direct_targets[:, np.newaxis]
    
    
    
    train_c_feat = get_compositional_features(train_structures, all_species)
    val_c_feat = get_compositional_features(val_structures, all_species)
    print('train targets shape: ', train_direct_targets.shape)
    print('train c feat shape: ', train_c_feat.shape)
    
    #print(np.mean(np.abs(val_direct_targets)))
    rgr = fit_with_nans(train_c_feat, train_direct_targets)
    train_direct_targets -= rgr.predict(train_c_feat)
    val_direct_targets -= rgr.predict(val_c_feat)
    #print(np.mean(np.abs(val_direct_targets)))
    np.save(f'results/{NAME_OF_CALCULATION}/self_contributions.npy', rgr.coef_)

train_molecules = [Molecule(structure, hypers.R_CUT, hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES, hypers.USE_TARGET_GRADS, hypers.TARGET_GRADS_NAME) for structure in tqdm(train_structures)]
val_molecules = [Molecule(structure, hypers.R_CUT, hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES, hypers.USE_TARGET_GRADS, hypers.TARGET_GRADS_NAME) for structure in tqdm(val_structures)]


molecules = train_molecules + val_molecules
max_nums = [molecule.get_max_num() for molecule in molecules]
max_num = np.max(max_nums)
print(max_num)

central_species = [molecule.central_species for molecule in molecules]
central_species = np.concatenate(central_species, axis = 0)

train_graphs = [molecule.get_graph(max_num, all_species) for molecule in tqdm(train_molecules)]
val_graphs = [molecule.get_graph(max_num, all_species) for molecule in tqdm(val_molecules)]


if hypers.USE_DIRECT_TARGETS:
    for index in range(len(train_structures)):
        now = train_direct_targets[index]
        now = now[None, :]
        train_graphs[index].y = torch.FloatTensor(now)
        

    for index in range(len(val_structures)):
        now = val_direct_targets[index]
        now = now[None, :]
        val_graphs[index].y = torch.FloatTensor(now)
    

    
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


import copy
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
if hypers.USE_DIRECT_TARGETS:
    direct_targets_logger = FullLogger()
    
if hypers.USE_TARGET_GRADS:
    target_grads_logger = FullLogger()



if hypers.USE_TARGET_GRADS:
    all_val_target_grads = []
    model.train(False)
    for batch in val_loader:
        if not hypers.MULTI_GPU:
            batch.to(device)
            model.augmentation = False
        else:
            model.module.augmentation = False
            
        _, _, _, targets_target_grads = model(batch)
        all_val_target_grads.append(targets_target_grads.data.cpu().numpy())
    all_val_target_grads = np.concatenate(all_val_target_grads, axis = 0)

    sliding_target_grads_rmse = get_rmse(all_val_target_grads, 0.0)
    
    target_grads_rmse_model_keeper = ModelKeeper()
    target_grads_mae_model_keeper = ModelKeeper()

if hypers.USE_DIRECT_TARGETS:
    sliding_direct_targets_rmse = get_rmse(val_direct_targets, np.mean(val_direct_targets))
    
    direct_targets_rmse_model_keeper = ModelKeeper()
    direct_targets_mae_model_keeper = ModelKeeper()

if hypers.USE_DIRECT_TARGETS and hypers.USE_TARGET_GRADS:
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
        #print(batch.y)
        
        #print("batch y shape: ", batch.y.shape)
        #print("batch y[0]: ", batch.y[0])
        predictions_direct_targets, targets_direct_targets, predictions_target_grads, targets_target_grads = model(batch)
        if hypers.USE_DIRECT_TARGETS:
            direct_targets_logger.train_logger.update(predictions_direct_targets, targets_direct_targets)
            loss_direct_targets = get_loss(predictions_direct_targets, targets_direct_targets)
        if hypers.USE_TARGET_GRADS:
            target_grads_logger.train_logger.update(predictions_target_grads, targets_target_grads)
            loss_target_grads = get_loss(predictions_target_grads, targets_target_grads)

        if hypers.USE_DIRECT_TARGETS and hypers.USE_TARGET_GRADS: 
            loss = hypers.DIRECT_TARGET_WEIGHT * loss_direct_targets / (sliding_direct_targets_rmse ** 2) + loss_target_grads / (sliding_target_grads_rmse ** 2)
            loss.backward()

        if hypers.USE_DIRECT_TARGETS and (not hypers.USE_TARGET_GRADS):
            loss_direct_targets.backward()
        if hypers.USE_TARGET_GRADS and (not hypers.USE_DIRECT_TARGETS):
            loss_target_grads.backward()


        optim.step()
        optim.zero_grad()

    model.train(False)
    for batch in val_loader:
        if not hypers.MULTI_GPU:
            batch.to(device)
            model.augmentation = False
        else:
            model.module.augmentation = False
            
        predictions_direct_targets, targets_direct_targets, predictions_target_grads, targets_target_grads = model(batch)
        if hypers.USE_DIRECT_TARGETS:
            direct_targets_logger.val_logger.update(predictions_direct_targets, targets_direct_targets)
        if hypers.USE_TARGET_GRADS:
            target_grads_logger.val_logger.update(predictions_target_grads, targets_target_grads)

    now = {}
    if hypers.USE_DIRECT_TARGETS:
        now['direct_targets'] = direct_targets_logger.flush()
    if hypers.USE_TARGET_GRADS:
        now['target_grads'] = target_grads_logger.flush()   
    now['lr'] = scheduler.get_last_lr()
    now['epoch'] = epoch
    now['elapsed_time'] = time.time() - TIME_SCRIPT_STARTED
    
    if hypers.USE_DIRECT_TARGETS:
        sliding_direct_targets_rmse = hypers.SLIDING_FACTOR * sliding_direct_targets_rmse + (1.0 - hypers.SLIDING_FACTOR) * now['direct_targets']['val']['rmse']

        direct_targets_mae_model_keeper.update(model, now['direct_targets']['val']['mae'], epoch)
        direct_targets_rmse_model_keeper.update(model, now['direct_targets']['val']['rmse'], epoch)


    if hypers.USE_TARGET_GRADS:
        sliding_target_grads_rmse = hypers.SLIDING_FACTOR * sliding_target_grads_rmse + (1.0 - hypers.SLIDING_FACTOR) * now['target_grads']['val']['rmse']
        target_grads_mae_model_keeper.update(model, now['target_grads']['val']['mae'], epoch)
        target_grads_rmse_model_keeper.update(model, now['target_grads']['val']['rmse'], epoch)    

    if hypers.USE_DIRECT_TARGETS and hypers.USE_TARGET_GRADS:
        multiplication_mae_model_keeper.update(model, now['target_grads']['val']['mae'] * now['direct_targets']['val']['mae'], epoch,
                                               additional_info = [now['direct_targets']['val']['mae'], now['target_grads']['val']['mae']])
        multiplication_rmse_model_keeper.update(model, now['target_grads']['val']['rmse'] * now['direct_targets']['val']['rmse'], epoch,
                                                additional_info = [now['direct_targets']['val']['rmse'], now['target_grads']['val']['rmse']])


    val_mae_message = "val mae/rmse:"
    train_mae_message = "train mae/rmse:"

    if hypers.USE_DIRECT_TARGETS:
        val_mae_message += f" {now['direct_targets']['val']['mae']}/{now['direct_targets']['val']['rmse']};"
        train_mae_message += f" {now['direct_targets']['train']['mae']}/{now['direct_targets']['train']['rmse']};"
    if hypers.USE_TARGET_GRADS:
        val_mae_message += f" {now['target_grads']['val']['mae']}/{now['target_grads']['val']['rmse']}"
        train_mae_message += f" {now['target_grads']['train']['mae']}/{now['target_grads']['train']['rmse']}"

    pbar.set_description(f"lr: {scheduler.get_last_lr()}; " + val_mae_message + train_mae_message)

    history.append(now)
    scheduler.step()
    elapsed = time.time() - TIME_SCRIPT_STARTED
    if hypers.MAX_TIME is not None:
        if elapsed > hypers.MAX_TIME:
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
if hypers.USE_DIRECT_TARGETS:    
    save_model('best_val_mae_direct_targets_model', direct_targets_mae_model_keeper)
    summary += f'best val mae in direct_targets: {direct_targets_mae_model_keeper.best_error} at epoch {direct_targets_mae_model_keeper.best_epoch}\n'
    
    save_model('best_val_rmse_direct_targets_model', direct_targets_rmse_model_keeper)
    summary += f'best val rmse in direct_targets: {direct_targets_rmse_model_keeper.best_error} at epoch {direct_targets_rmse_model_keeper.best_epoch}\n'
    
if hypers.USE_TARGET_GRADS:
    save_model('best_val_mae_target_grads_model', target_grads_mae_model_keeper)
    summary += f'best val mae in target_grads: {target_grads_mae_model_keeper.best_error} at epoch {target_grads_mae_model_keeper.best_epoch}\n'
    
    save_model('best_val_rmse_target_grads_model', target_grads_rmse_model_keeper)
    summary += f'best val rmse in target_grads: {target_grads_rmse_model_keeper.best_error} at epoch {target_grads_rmse_model_keeper.best_epoch}\n'
    
if hypers.USE_DIRECT_TARGETS and hypers.USE_TARGET_GRADS:
    save_model('best_val_mae_both_model', multiplication_mae_model_keeper)
    summary += f'best both (multiplication) mae in direct_targets: {multiplication_mae_model_keeper.additional_info[0]} in target_grads: {multiplication_mae_model_keeper.additional_info[1]} at epoch {multiplication_mae_model_keeper.best_epoch}\n'
    
    
    save_model('best_val_rmse_both_model', multiplication_rmse_model_keeper)
    summary += f'best both (multiplication) rmse in direct_targets: {multiplication_rmse_model_keeper.additional_info[0]} in target_grads: {multiplication_rmse_model_keeper.additional_info[1]} at epoch {multiplication_rmse_model_keeper.best_epoch}\n'
    
with open(f"results/{NAME_OF_CALCULATION}/summary.txt", 'w') as f:
    print(summary, file = f)
    


    

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
from torch_geometric.nn import DataParallel

from molecule import Molecule, batch_to_dict
from hypers import Hypers
from pet import PET
from utilities import FullLogger
from utilities import get_rmse, get_mae, get_relative_rmse, get_loss
from analysis import get_structural_batch_size, convert_atomic_throughput


STRUCTURES_PATH = sys.argv[1]
HYPERS_PATH = sys.argv[2]
PATH_TO_MODEL_STATE_DICT = sys.argv[3]
ALL_SPECIES_PATH = sys.argv[4]
N_AUG = int(sys.argv[5])

'''STRUCTURES_PATH = 'small_data/test_small.xyz'
HYPERS_PATH = 'results/test_calc_continuation_0/hypers_used.yaml'
ALL_SPECIES_PATH = 'results/test_calc_continuation_0/all_species.npy'
SELF_CONTRIBUTIONS_PATH = 'results/test_calc_continuation_0/self_contributions.npy'
PATH_TO_MODEL_STATE_DICT = 'results/test_calc_continuation_0/best_val_mae_energies_model_state_dict' '''

hypers = Hypers()
hypers.load_from_file(HYPERS_PATH)
structures = ase.io.read(STRUCTURES_PATH, index = ':')

all_species = np.load(ALL_SPECIES_PATH)

molecules = [Molecule(structure, hypers.R_CUT, False) for structure in tqdm(structures)]
max_nums = [molecule.get_max_num() for molecule in molecules]
max_num = np.max(max_nums)
graphs = [molecule.get_graph(max_num, all_species) for molecule in tqdm(molecules)]

if hypers.MULTI_GPU:
    loader = DataListLoader(graphs, batch_size=hypers.STRUCTURAL_BATCH_SIZE, shuffle=False)
else:        
    loader = DataLoader(graphs, batch_size=hypers.STRUCTURAL_BATCH_SIZE, shuffle=False)

add_tokens = []
for _ in range(hypers.N_GNN_LAYERS - 1):
    add_tokens.append(hypers.ADD_TOKEN_FIRST)
add_tokens.append(hypers.ADD_TOKEN_SECOND)

model = PET(hypers, hypers.TRANSFORMER_D_MODEL, hypers.TRANSFORMER_N_HEAD,
                       hypers.TRANSFORMER_DIM_FEEDFORWARD, hypers.N_TRANS_LAYERS, 
                       0.0, len(all_species), 
                       hypers.N_GNN_LAYERS, hypers.HEAD_N_NEURONS, hypers.TRANSFORMERS_CENTRAL_SPECIFIC, hypers.HEADS_CENTRAL_SPECIFIC, 
                       add_tokens).cuda()
if hypers.MULTI_GPU:
    model = DataParallel(model)
    device = torch.device('cuda:0')
    model = model.to(device)
    
model.load_state_dict(torch.load(PATH_TO_MODEL_STATE_DICT))
model.eval()


dipoles_ground_truth = np.array([struc.info['dipole_b3lyp'] for struc in structures])
   

all_dipoles_predicted = []
    
for _ in tqdm(range(N_AUG)):
    dipoles_predicted = []
    for batch in loader:
        if not hypers.MULTI_GPU:
            batch.cuda()
            model.augmentation = True
        else:
            model.module.augmentation = True
            
        predictions_dipoles, targets_dipoles, predictions_forces, targets_forces = model(batch)
       
        dipoles_predicted.append(predictions_dipoles.data.cpu().numpy())
       
            
    
    dipoles_predicted = np.concatenate(dipoles_predicted, axis = 0)
    all_dipoles_predicted.append(dipoles_predicted)
        
   
 
all_dipoles_predicted = [el[np.newaxis] for el in all_dipoles_predicted]
all_dipoles_predicted = np.concatenate(all_dipoles_predicted, axis = 0)
dipoles_predicted_mean = np.mean(all_dipoles_predicted, axis = 0)
    

print(f"dipoles mae per component: {get_mae(dipoles_ground_truth, dipoles_predicted_mean)}")
print(f"dipoles rmse per component: {get_rmse(dipoles_ground_truth, dipoles_predicted_mean)}")
    

def get_mae_absolute_value(predictions, targets):
    #print(predictions.shape, targets.shape)
    predictions_abs = np.sqrt(np.sum(predictions ** 2, axis = 1))
    targets_abs = np.sqrt(np.sum(targets ** 2, axis = 1))
    #print(predictions_abs.shape)
    delta = predictions_abs - targets_abs
    return np.mean(np.abs(delta))    
    
print(f"dipoles mae absolute value: { get_mae_absolute_value(dipoles_predicted_mean, dipoles_ground_truth)}")    


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


from sp_frames_calculator import SPFramesCalculator
from pet_sp import PETSP
np.random.seed(0)

EPSILON = 1e-10

STRUCTURES_PATH = sys.argv[1]
PATH_TO_CALC_FOLDER_MAIN = sys.argv[2]
CHECKPOINT_MAIN = sys.argv[3]

PATH_TO_CALC_FOLDER_AUX = sys.argv[4]
CHECKPOINT_AUX = sys.argv[5]

bool_map = {'True' : True, 'False' : False}

SP_HYPERS_PATH = sys.argv[6]
DEFAULT_HYPERS_PATH = sys.argv[7]

BATCH_SIZE_SP = int(sys.argv[8])
PATH_SAVE_PREDICTIONS = sys.argv[9]
SHOW_PROGRESS = bool_map[sys.argv[10]]
MAX_NUM = sys.argv[11]

if MAX_NUM == 'None' or MAX_NUM == 'none':
    MAX_NUM = None
else:
    MAX_NUM = int(sys.argv[11])
    
def load_model(path_to_calc_folder, checkpoint):
    hypers_path = path_to_calc_folder + '/hypers_used.yaml'
    path_to_model_state_dict = path_to_calc_folder + '/' + checkpoint + '_state_dict'
    all_species_path = path_to_calc_folder + '/all_species.npy'
   
    
    hypers = Hypers()
    
    # loading default values for the new hypers potentially added into the codebase after the calculation is done
    # assuming that the default values do not change the logic
    hypers.set_from_files(hypers_path, DEFAULT_HYPERS_PATH, check_dublicated = False)
    
    all_species = np.load(all_species_path)
    
    
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

        model.load_state_dict(torch.load(path_to_model_state_dict))
        model.eval()

        model = model.module
    else:
        model.load_state_dict(torch.load(path_to_model_state_dict))
        model.eval()
        
    return model, hypers, all_species
    
model_main, hypers_main, all_species_main= load_model(PATH_TO_CALC_FOLDER_MAIN, CHECKPOINT_MAIN)



if PATH_TO_CALC_FOLDER_AUX == 'None' or PATH_TO_CALC_FOLDER_AUX == 'none':
    PATH_TO_CALC_FOLDER_AUX = None
    
if PATH_TO_CALC_FOLDER_AUX is None:
    model_aux, hypers_aux, all_species_aux = None, None, None
    
    USE_ADDITIONAL_SCALAR_ATTRIBUTES_DATA = hypers_main.USE_ADDITIONAL_SCALAR_ATTRIBUTES    
    
    if hypers_main.MULTI_GPU:
        print("using only 1 gpu, multigpu for sp is not implemented")
        
else:
    model_aux, hypers_aux, all_species_aux = load_model(PATH_TO_CALC_FOLDER_AUX, CHECKPOINT_AUX)

   

    if np.abs(hypers_aux.R_CUT - hypers_main.R_CUT) > EPSILON:
        raise ValueError("R_CUT of main and aux models should be same in the current implementation")
        
    if not are_same(all_species_main, all_species_aux):
        raise ValueError("all species should be same")
        
    USE_ADDITIONAL_SCALAR_ATTRIBUTES_DATA = hypers_main.USE_ADDITIONAL_SCALAR_ATTRIBUTES or hypers_aux.USE_ADDITIONAL_SCALAR_ATTRIBUTES
  
    if hypers_main.MULTI_GPU or hypers_aux.MULTI_GPU:
        print("using only 1 gpu, multigpu for sp is not implemented")

        
R_CUT = hypers_main.R_CUT
all_species = all_species_main
    


structures = ase.io.read(STRUCTURES_PATH, index = ':')

molecules = [Molecule(structure, R_CUT, USE_ADDITIONAL_SCALAR_ATTRIBUTES_DATA) for structure in tqdm(structures)]
max_nums = [molecule.get_max_num() for molecule in molecules]
max_num = np.max(max_nums)
graphs = [molecule.get_graph(max_num, all_species) for molecule in tqdm(molecules)]


       
loader = DataLoader(graphs, 1, shuffle=False)


dipoles_ground_truth = np.array([struc.info['dipole_b3lyp'] for struc in structures])
 

sp_hypers = Hypers()
sp_hypers.load_from_file(SP_HYPERS_PATH)
sp_frames_calculator = SPFramesCalculator(sp_hypers)


    
model_sp = PETSP(model_main, model_aux, R_CUT, sp_frames_calculator, BATCH_SIZE_SP, 
                 epsilon = EPSILON, show_progress = SHOW_PROGRESS, max_num = MAX_NUM,
                n_aug = sp_hypers.N_ADDITIONAL_AUG).cuda()


all_dipoles_predicted = []
dipoles_predicted = []


#print("len loader: ", len(loader), len(molecules))

n_frames_used, aux_weights, total_main_weights  = [], [], []
for batch in tqdm(loader):
    #print(batch)
    batch.cuda()
    #with torch.autograd.set_detect_anomaly(True):
    n_frames, aux_weight, total_main_weight, predictions_dipoles, targets_dipoles = model_sp(batch)
    n_frames_used.append(n_frames)
    if isinstance(total_main_weight, float):
        total_main_weights.append(total_main_weight)
    else:
        total_main_weights.append(total_main_weight.data.cpu().numpy())
        
    aux_weights.append(float(aux_weight.data.cpu().numpy()))
   
    dipoles_predicted.append(predictions_dipoles)
    

dipoles_predicted = np.concatenate(dipoles_predicted, axis = 0)
all_dipoles_predicted.append(dipoles_predicted)
        
        

all_dipoles_predicted = [el[np.newaxis] for el in all_dipoles_predicted]
all_dipoles_predicted = np.concatenate(all_dipoles_predicted, axis = 0)
dipoles_predicted_mean = np.mean(all_dipoles_predicted, axis = 0)

    
print("Average number of active coordinate systems: ", np.mean(n_frames_used))
#print("aux_weights: ", aux_weights)
n_fully_aux, n_partially_aux = 0, 0
for weight in aux_weights:
    if weight > EPSILON:
        n_partially_aux += 1
        
#print(total_main_weights)
for weight in total_main_weights:
    if weight < EPSILON:
        n_fully_aux += 1
            
#print("The number of structures handled completely by auxiliary model is: ", n_fully_aux, '; ratio is', n_fully_aux / len(aux_weights))
if PATH_TO_CALC_FOLDER_AUX is not None:
    print(f"Auxiliary model was active for {n_partially_aux}/{len(aux_weights)} structures ")


   
    
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
    
    
if (PATH_SAVE_PREDICTIONS != 'None') and (PATH_SAVE_PREDICTIONS != 'none'):
    np.save(PATH_SAVE_PREDICTIONS + '/dipoles_predicted.npy', dipoles_predicted_mean)
    

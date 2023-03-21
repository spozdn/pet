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

EPSILON = 1e-10

STRUCTURES_PATH = sys.argv[1]
PATH_TO_CALC_FOLDER_MAIN = sys.argv[2]
CHECKPOINT_MAIN = sys.argv[3]

PATH_TO_CALC_FOLDER_AUX = sys.argv[4]
CHECKPOINT_AUX = sys.argv[5]


bool_map = {'True' : True, 'False' : False}
USE_AUG = bool_map[sys.argv[6]]
N_AUG = int(sys.argv[7])
SP_HYPERS_PATH = sys.argv[8]


def load_model(path_to_calc_folder, checkpoint):
    hypers_path = path_to_calc_folder + '/hypers_used.yaml'
    path_to_model_state_dict = path_to_calc_folder + '/' + checkpoint + '_state_dict'
    all_species_path = path_to_calc_folder + '/all_species.npy'
    self_contributions_path = path_to_calc_folder + '/self_contributions.npy'
    
    hypers = Hypers()
    hypers.load_from_file(hypers_path)
    
    all_species = np.load(all_species_path)
    if hypers.USE_ENERGIES:
        self_contributions = np.load(self_contributions_path)
    
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
        
    return model, hypers, all_species, self_contributions
    
model_main, hypers_main, all_species_main, self_contributions_main = load_model(PATH_TO_CALC_FOLDER_MAIN, CHECKPOINT_MAIN)
model_aux, hypers_aux, all_species_aux, self_contributions_aug = load_model(PATH_TO_CALC_FOLDER_AUX, CHECKPOINT_AUX)

print("self contributions main", self_contributions_main)
print("self contributions aug", self_contributions_aug)
print("self contributions delta", self_contributions_main - self_contributions_aug)

if np.abs(hypers_aux.R_CUT - hypers_main.R_CUT) > EPSILON:
    raise ValueError("R_CUT of main and aug models should be same in the current implementation")

R_CUT = hypers_main.R_CUT

def are_same(first, second):
    if len(first) != len(second):
        return False
    
    for i in range(len(first)):
        if abs(first[i] - second[i]) > EPSILON:
            return False
    return True

if are_same(self_contributions_main, self_contributions_aug):
    self_contributions = self_contributions_main
else:
    raise ValueError("self contributions should be same (in this rudementary implementation)")
    
if are_same(all_species_main, all_species_aux):
    all_species = all_species_main
else:
    raise ValueError("all species should be same")

USE_ADDITIONAL_SCALAR_ATTRIBUTES_DATA = hypers_main.USE_ADDITIONAL_SCALAR_ATTRIBUTES or hypers_aux.USE_ADDITIONAL_SCALAR_ATTRIBUTES

USE_FORCES = hypers_main.USE_FORCES and hypers_aux.USE_FORCES
USE_ENERGIES = hypers_main.USE_ENERGIES and hypers_aux.USE_ENERGIES

structures = ase.io.read(STRUCTURES_PATH, index = ':')

molecules = [Molecule(structure, R_CUT, USE_ADDITIONAL_SCALAR_ATTRIBUTES_DATA, USE_FORCES) for structure in tqdm(structures)]
max_nums = [molecule.get_max_num() for molecule in molecules]
max_num = np.max(max_nums)
graphs = [molecule.get_graph(max_num, all_species) for molecule in tqdm(molecules)]

if hypers_main.MULTI_GPU or hypers_aux.MULTI_GPU:
    print("using only 1 gpu, multigpu for sp is not implemented")
       
loader = DataLoader(graphs, 1, shuffle=False)

if USE_ENERGIES:
    energies_ground_truth = np.array([struc.info['energy'] for struc in structures])
    
if USE_FORCES:
    forces_ground_truth = [struc.arrays['forces'] for struc in structures]
    forces_ground_truth = np.concatenate(forces_ground_truth, axis = 0)
    
    
class PETSP(torch.nn.Module):
    def __init__(self, model_main, model_aux, r_cut, use_energies, use_forces, sp_frames_calculator, additional_rotations = None):
        super(PETSP, self).__init__()
        self.r_cut = r_cut
        self.use_energies = use_energies
        self.use_forces = use_forces
        
        self.model_main = model_main
        self.model_aux = model_aux
        self.model_main.task = 'energies'
        self.model_aux.task = 'energies'
        
        
        self.sp_frames_calculator = sp_frames_calculator
        self.task = 'both'
        self.additional_rotations = additional_rotations
        if self.additional_rotations is None:
            self.additional_rotations = [torch.eye(3)]
        
    def get_all_frames(self, batch):
        all_envs = []
        for env_index in range(batch.x.shape[0]):
            mask_now = torch.logical_not(batch.mask[env_index])
            env_now = batch.x[env_index][mask_now]
            all_envs.append(env_now)
            
        r_cut = torch.tensor(self.r_cut, device = batch.x.device)
        return self.sp_frames_calculator.get_all_frames_global(all_envs, r_cut)
    
    def forward(self, batch):        
        if self.task == 'both':
            return self.get_targets(batch)        
        
        frames, weights, weight_aux = self.get_all_frames(batch)
        #print('weight aux: ', type(weight_aux), weight_aux)
        batch.x_initial = batch.x
        #print(len(frames))
        predictions_accumulated = 0.0
        weight_accumulated = 0.0
        
        for additional_rotation in self.additional_rotations:
            additional_rotation = additional_rotation.to(batch.x.device)
            for frame, weight in zip(frames, weights):
                frame = torch.matmul(additional_rotation, frame)
                frame = frame[None]
                frame = frame.repeat(batch.x_initial.shape[0], 1, 1)
                batch.x = torch.bmm(batch.x_initial, frame)
                predictions_now = self.model_main(batch)
                #print(predictions_now)
                predictions_accumulated = predictions_accumulated + predictions_now * weight
                weight_accumulated += weight
                
        if weight_aux > EPSILON:
            weight_accumulated += weight_aux
            predictions_accumulated += self.model_aux(batch) * weight_aux
        
        return predictions_accumulated / weight_accumulated, len(frames), weight_aux
    
    def get_targets(self, batch):
        
        batch.x_initial = batch.x.clone().detach()
        batch.x_initial.requires_grad = True
        batch.x = batch.x_initial
        
        self.task = 'energies'
        predictions, n_frames, weight_aux = self(batch)
        self.task = 'both'
        if self.use_forces:
            grads  = torch.autograd.grad(predictions, batch.x_initial, grad_outputs = torch.ones_like(predictions),
                                    create_graph = True)[0]
            neighbors_index = batch.neighbors_index.transpose(0, 1)
            neighbors_pos = batch.neighbors_pos
            grads_messaged = grads[neighbors_index, neighbors_pos]
            grads[batch.mask] = 0.0
            first = grads.sum(dim = 1)
            grads_messaged[batch.mask] = 0.0
            second = grads_messaged.sum(dim = 1)
        
        result = [n_frames, weight_aux]
        if self.use_energies:
            result.append(predictions)
            result.append(batch.y)
        else:
            result.append(None)
            result.append(None)
            
        if  self.use_forces:
            result.append(first - second)
            result.append(batch.forces)
        else:
            result.append(None)
            result.append(None)
            
        return result
    

sp_hypers = Hypers()
sp_hypers.load_from_file(SP_HYPERS_PATH)
sp_frames_calculator = SPFramesCalculator(sp_hypers)

if USE_AUG:
    additional_rotations = [torch.FloatTensor(el) for el in Rotation.random(N_AUG).as_matrix()]
else:
    additional_rotations = None
    
model_sp = PETSP(model_main, model_aux, R_CUT, USE_ENERGIES, USE_FORCES, sp_frames_calculator, additional_rotations = additional_rotations).cuda()
if USE_ENERGIES:
    all_energies_predicted = []
    
if USE_FORCES:
    all_forces_predicted = []
    
if USE_ENERGIES:
    energies_predicted = []
if USE_FORCES:
    forces_predicted = []

#print("len loader: ", len(loader), len(molecules))

n_frames_used, aux_weights = [], []
for batch in tqdm(loader):
    #print(batch)
    batch.cuda()
    n_frames, aux_weight, predictions_energies, targets_energies, predictions_forces, targets_forces = model_sp(batch)
    n_frames_used.append(n_frames)
    aux_weights.append(float(aux_weight.data.cpu().numpy()))
    if USE_ENERGIES:
        energies_predicted.append(predictions_energies.data.cpu().numpy())
    if USE_FORCES:
        forces_predicted.append(predictions_forces.data.cpu().numpy())

if USE_ENERGIES:
    energies_predicted = np.concatenate(energies_predicted, axis = 0)
    all_energies_predicted.append(energies_predicted)

if USE_FORCES:
    forces_predicted = np.concatenate(forces_predicted, axis = 0)
    all_forces_predicted.append(forces_predicted)
        
        
if USE_ENERGIES:
    all_energies_predicted = [el[np.newaxis] for el in all_energies_predicted]
    all_energies_predicted = np.concatenate(all_energies_predicted, axis = 0)
    energies_predicted_mean = np.mean(all_energies_predicted, axis = 0)
    
if USE_FORCES:
    all_forces_predicted = [el[np.newaxis] for el in all_forces_predicted]
    all_forces_predicted = np.concatenate(all_forces_predicted, axis = 0)
    forces_predicted_mean = np.mean(all_forces_predicted, axis = 0)

    
print("Average number of active coordinate systems: ", np.mean(n_frames_used))
print("aux_weights: ", aux_weights)
n_fully_aux, n_partially_aux = 0, 0
for weight in aux_weights:
    if np.abs(weight - 1.0) < EPSILON:
        n_fully_aux += 1
    else:
        if weight > EPSILON:
            n_partially_aux += 1
            
print("The number of structures handled completely by auxiliary model is: ", n_fully_aux, ';', n_fully_aux / len(aux_weights))
print("The number of structures handled partially by auxiliary model is: ", n_partially_aux, ';', n_partially_aux / len(aux_weights))

if USE_ENERGIES:
    compositional_features = get_compositional_features(structures, all_species)
    self_contributions_energies = []
    for i in range(len(structures)):
        self_contributions_energies.append(np.dot(compositional_features[i], self_contributions))
    self_contributions_energies = np.array(self_contributions_energies)
    
    energies_predicted_mean = energies_predicted_mean + self_contributions_energies
    
    print(f"energies mae: {get_mae(energies_ground_truth, energies_predicted_mean)}")
    print(f"energies rmse: {get_rmse(energies_ground_truth, energies_predicted_mean)}")
    
if USE_FORCES:
    print(f"forces mae per component: {get_mae(forces_ground_truth, forces_predicted_mean)}")
    print(f"forces rmse per component: {get_rmse(forces_ground_truth, forces_predicted_mean)}")
    

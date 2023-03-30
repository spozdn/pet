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


class PETSP(torch.nn.Module):
    def __init__(self, model_main, model_aux, r_cut, use_energies, use_forces, sp_frames_calculator, batch_size_sp, additional_rotations = None, epsilon = 1e-10):
        super(PETSP, self).__init__()
        self.r_cut = r_cut
        self.use_energies = use_energies
        self.use_forces = use_forces
        
        self.model_main = model_main
        self.model_aux = model_aux
        self.model_main.task = 'energies'
        if self.model_aux is not None:
            self.model_aux.task = 'energies'
        
        
        self.sp_frames_calculator = sp_frames_calculator
        self.batch_size_sp = batch_size_sp
        self.additional_rotations = additional_rotations
        self.epsilon = epsilon
        if self.additional_rotations is None:
            self.additional_rotations = [torch.eye(3)]
        
    def get_all_frames(self, batch):
        all_envs = []
        for env_index in range(batch.x.shape[0]):
            mask_now = torch.logical_not(batch.mask[env_index])
            env_now = batch.x[env_index][mask_now]
            all_envs.append(env_now)
            
        return self.sp_frames_calculator.get_all_frames_global(all_envs, self.r_cut)
    
    def get_all_contributions(self, batch):
        x_initial = batch.x
        x_initial.requires_grad = True
        
        batch.x = x_initial
        frames, weights, weight_aux = self.get_all_frames(batch)
        
        weight_accumulated = 0.0
        for weight in weights:
            weight_accumulated = weight_accumulated + weight
            
        total_main_weight = weight_accumulated
        weight_accumulated = weight_accumulated * len(self.additional_rotations)
        weight_accumulated = weight_accumulated + weight_aux
        
        predictions_accumulated = 0.0
        num_handled = 0
        for additional_rotation in self.additional_rotations:
            additional_rotation = additional_rotation.to(batch.x.device)
            for index in range(len(frames)):
                frame = frames[index]
                weight = weights[index]
                #print(x_initial[0, 0, 0], batch.x[0, 0, 0])
                frame = torch.matmul(frame, additional_rotation)
                frame = frame[None]
                frame = frame.repeat(x_initial.shape[0], 1, 1)
                batch.x = torch.bmm(x_initial, frame)
                predictions_now = self.model_main(batch)
                #print(predictions_now)
                predictions_accumulated = predictions_accumulated + predictions_now * weight
                
                num_handled += 1
                if num_handled == self.batch_size_sp:
        
                    result = predictions_accumulated / weight_accumulated
                    result.backward()
                    grads = x_initial.grad
                    x_initial.grad = None
                    yield result, grads, len(frames), weight_aux, total_main_weight
                    
                    batch.x = x_initial
                    frames, weights, weight_aux = self.get_all_frames(batch)
                    
                    weight_accumulated = 0.0
                    for weight in weights:
                        weight_accumulated = weight_accumulated + weight
                    weight_accumulated = weight_accumulated * len(self.additional_rotations)
                    weight_accumulated = weight_accumulated + weight_aux
                    
                    predictions_accumulated = 0.0
                    num_handled = 0
                
        if weight_aux > self.epsilon:
            if self.model_aux is not None:
                batch.x = x_initial
                predictions_accumulated = predictions_accumulated + self.model_aux(batch) * weight_aux
                num_handled += 1
            
        if num_handled > 0:
            
            result = predictions_accumulated / weight_accumulated
            
            result.backward()
            grads = x_initial.grad
            x_initial.grad = None
                    
            yield result, grads, len(frames), weight_aux, total_main_weight
            
    def forward(self, batch):
        predictions_total, forces_predicted_total = 0.0, 0.0
        n_frames = None
        for predictions, grads, n_frames, weight_aux, total_main_weight in self.get_all_contributions(batch):
            predictions_total += predictions
            if self.use_forces:
                neighbors_index = batch.neighbors_index.transpose(0, 1)
                neighbors_pos = batch.neighbors_pos
                grads_messaged = grads[neighbors_index, neighbors_pos]
                grads[batch.mask] = 0.0
                first = grads.sum(dim = 1)
                grads_messaged[batch.mask] = 0.0
                second = grads_messaged.sum(dim = 1)
                forces_predicted = first - second
                forces_predicted_total += forces_predicted
            
        if n_frames is None:
            raise ValueError("all collinear problem happened, but aux model was not provided")
        
        result = [n_frames, weight_aux, total_main_weight]
        if self.use_energies:
            result.append(predictions_total)
            result.append(batch.y)
        else:
            result.append(None)
            result.append(None)
            
        if self.use_forces:
            result.append(forces_predicted_total)
            result.append(batch.forces)
        else:
            result.append(None)
            result.append(None)
           
        return result
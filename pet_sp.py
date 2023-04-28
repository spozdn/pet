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
from torch_geometric.data import Batch

from sp_frames_calculator import SPFramesCalculator


class PETSP(torch.nn.Module):
    def __init__(self, model_main, model_aux, r_cut, sp_frames_calculator, batch_size_sp,
                 epsilon = 1e-10, show_progress = False, max_num = None, n_aug = None):
        super(PETSP, self).__init__()
        self.show_progress = show_progress
        self.r_cut = r_cut
       
        self.n_aug = n_aug
        
        self.max_num = max_num
        self.model_main = model_main
        self.model_aux = model_aux
        self.model_main.task = 'dipoles'
        if self.model_aux is not None:
            self.model_aux.task = 'dipoles'
        
        
        self.sp_frames_calculator = sp_frames_calculator
        self.batch_size_sp = batch_size_sp
        
        self.epsilon = epsilon
        
    def get_all_frames(self, batch):
        all_envs = []
        for env_index in range(batch.x.shape[0]):
            mask_now = torch.logical_not(batch.mask[env_index])
            env_now = batch.x[env_index][mask_now]
            all_envs.append(env_now)
            
        return self.sp_frames_calculator.get_all_frames_global(all_envs, self.r_cut)
    
    def get_all_contributions(self, batch, additional_rotations):
        x_initial = batch.x
       
        batch.x = x_initial
        frames, weights, weight_aux = self.get_all_frames(batch)
        if self.max_num is not None:
            if len(frames) > self.max_num:
                raise ValueError(f"number of frames ({len(frames)}) is bigger than the upper bound provided")
                
        if self.show_progress:
            print("number of frames now: ", len(frames))
        weight_accumulated = 0.0
        for weight in weights:
            weight_accumulated = weight_accumulated + weight
            
        total_main_weight = weight_accumulated
        weight_accumulated = weight_accumulated * len(additional_rotations)
        if self.model_aux is not None:
            weight_accumulated = weight_accumulated + weight_aux
        
        
        strucs_minibatch_sp = []
        weights_minibatch_sp = []
        frames_minibatch_sp = []
        num_handled = 0
        for additional_rotation in additional_rotations:
            additional_rotation = additional_rotation.to(batch.x.device)
            for index in range(len(frames)):
                frame = frames[index]
                weight = weights[index]
                #print(x_initial[0, 0, 0], batch.x[0, 0, 0])
                frame = torch.matmul(frame, additional_rotation)
                frames_minibatch_sp.append(frame[None])
                frame = frame[None]
                frame = frame.repeat(x_initial.shape[0], 1, 1)
                
                batch_now = batch.clone()
                batch_now.x = torch.bmm(x_initial, frame)
                
                strucs_minibatch_sp.append(batch_now)
                weights_minibatch_sp.append(weight[None])
                
                
                num_handled += 1
                if num_handled == self.batch_size_sp:
                    
                    batch_sp = Batch.from_data_list(strucs_minibatch_sp)
                    weights_minibatch_sp = torch.cat(weights_minibatch_sp)
                    frames_minibatch_sp = torch.cat(frames_minibatch_sp, dim = 0)
                    inverse_frames_minibatch_sp = frames_minibatch_sp.transpose(1, 2)
                    
                    predictions = self.model_main(batch_sp)
                    
                    predictions = predictions[:, None, :]
                    predictions = torch.bmm(predictions, inverse_frames_minibatch_sp)
                    predictions = predictions[:, 0, :]
            
            
                    #print("shapes: ", predictions.shape, weights_minibatch_sp.shape, frames_minibatch_sp.shape)
                    predictions_accumulated = torch.sum(predictions * weights_minibatch_sp[:, None], dim = 0)[None, :]
                    
                    result = predictions_accumulated / weight_accumulated
                   
                    yield result, len(frames), weight_aux, total_main_weight
                    
                    
                    frames, weights, weight_aux = self.get_all_frames(batch)
                    
                    weight_accumulated = 0.0
                    for weight in weights:
                        weight_accumulated = weight_accumulated + weight
                    weight_accumulated = weight_accumulated * len(additional_rotations)
                    if self.model_aux is not None:
                        weight_accumulated = weight_accumulated + weight_aux
                    
                    strucs_minibatch_sp = []
                    weights_minibatch_sp = []
                    frames_minibatch_sp = []
                    num_handled = 0
        
        if num_handled > 0:
            
            batch_sp = Batch.from_data_list(strucs_minibatch_sp)
            weights_minibatch_sp = torch.cat(weights_minibatch_sp)
            frames_minibatch_sp = torch.cat(frames_minibatch_sp, dim = 0)
            inverse_frames_minibatch_sp = frames_minibatch_sp.transpose(1, 2)
            
            
            predictions = self.model_main(batch_sp)
            predictions = predictions[:, None, :]
            predictions = torch.bmm(predictions, inverse_frames_minibatch_sp)
            predictions = predictions[:, 0, :]
                    
            #print("shapes: ", predictions.shape, weights_minibatch_sp.shape)
            predictions_accumulated = torch.sum(predictions * weights_minibatch_sp[:, None], dim = 0)[None, :]
                    
            result = predictions_accumulated / weight_accumulated
           
                    
            yield result, len(frames), weight_aux, total_main_weight
            
        if weight_aux > self.epsilon:
            if self.model_aux is not None:
                
                frames, weights, weight_aux = self.get_all_frames(batch)
                    
                weight_accumulated = 0.0
                for weight in weights:
                    weight_accumulated = weight_accumulated + weight
                weight_accumulated = weight_accumulated * len(additional_rotations)
                if self.model_aux is not None:
                    weight_accumulated = weight_accumulated + weight_aux
                        
                        
                batch.x = x_initial
                predictions_accumulated = self.model_aux(batch) * weight_aux
                
                result = predictions_accumulated / weight_accumulated
               
                yield result, len(frames), weight_aux, total_main_weight
            
        
            
    def forward(self, batch):
        if self.n_aug is None:
            additional_rotations = [torch.eye(3)]
        else:
            additional_rotations = [torch.FloatTensor(el) for el in Rotation.random(self.n_aug).as_matrix()]
            
        predictions_total = 0.0
        n_frames = None
        for predictions, n_frames, weight_aux, total_main_weight in tqdm(self.get_all_contributions(batch, additional_rotations), disable = not self.show_progress):
            predictions_total += predictions
            
        if n_frames is None:
            raise ValueError("all collinear problem happened, but aux model was not provided")
        
        result = [n_frames, weight_aux, total_main_weight]
       
        result.append(predictions_total)
        result.append(batch.y)
       
                   
        return result
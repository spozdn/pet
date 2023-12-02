from .utilities import get_all_species, get_compositional_features
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
from .utilities import ModelKeeper
import time
from scipy.spatial.transform import Rotation
from torch.optim.lr_scheduler import LambdaLR
import sys
import copy
import inspect
import yaml
from torch_geometric.nn import DataParallel
import random
from .molecule import Molecule, batch_to_dict
from .hypers import Hypers
from .pet import PET
from .utilities import FullLogger
from .utilities import get_rmse, get_mae, get_relative_rmse, get_loss
from .analysis import get_structural_batch_size, convert_atomic_throughput


class SingleStructCalculator():
    def __init__(self, path_to_calc_folder, checkpoint="best_val_rmse_both_model", default_hypers_path="default_hypers.yaml", device="cpu"): 
        hypers_path = path_to_calc_folder + '/hypers_used.yaml'
        path_to_model_state_dict = path_to_calc_folder + '/' + checkpoint + '_state_dict'
        all_species_path = path_to_calc_folder + '/all_species.npy'
        self_contributions_path = path_to_calc_folder + '/self_contributions.npy'
        
        hypers = Hypers()
        hypers.set_from_files(hypers_path, default_hypers_path, check_duplicated = False)
        
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
                               add_tokens).to(device)

        if hypers.MULTI_GPU and torch.cuda.is_available():
            model = DataParallel(model)
            model = model.to( torch.device('cuda:0'))

        model.load_state_dict(torch.load(path_to_model_state_dict, map_location=torch.device(device)))
        model.eval()
        
        self.model = model
        self.hypers = hypers
        self.all_species = all_species
        self.model.augmentation = False
        
    def forward(self, structure):
        molecule = Molecule(structure, self.hypers.R_CUT, 
                            self.hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
                            self.hypers.USE_FORCES)
        
        graph = molecule.get_graph(molecule.get_max_num(), self.all_species)
        graph.y = 0
        graph.forces = np.zeros_like(structure.positions)
        prediction_energy, _, prediction_forces, _ = self.model(graph)
        print("PET Predictions ", prediction_energy, prediction_forces)
        return prediction_energy.data.cpu().numpy(), prediction_forces.data.cpu().numpy()
        
        
        
        

import torch
import numpy as np

from scipy.spatial.transform import Rotation
import copy


def get_all_species(structures):
    
    all_species = []
    for structure in structures:
        all_species.append(np.array(structure.get_atomic_numbers()))
    all_species = np.concatenate(all_species, axis=0)
    all_species = np.sort(np.unique(all_species))
    return all_species


def get_compositional_features(structures, all_species):
    
    result = np.zeros([len(structures), len(all_species)])
    for i, structure in enumerate(structures):
        species_now = structure.get_atomic_numbers()
        for j, specie in enumerate(all_species):
            num = np.sum(species_now == specie)
            result[i, j] = num
    return result

def get_length(delta):
    return np.sqrt(np.sum(delta * delta))

class ModelKeeper:
    def __init__(self):
        self.best_model = None
        self.best_error = None
        self.best_epoch = None
        self.additional_info = None
        
    def update(self, model_now, error_now, epoch_now, additional_info = None):
        if (self.best_error is None) or (error_now < self.best_error):
            self.best_error = error_now
            self.best_model = copy.deepcopy(model_now)
            self.best_epoch = epoch_now
            self.additional_info = additional_info
            
class Logger:
    def __init__(self):
        self.predictions = []
        self.targets = []
        
    def update(self, predictions_now, targets_now):
        self.predictions.append(predictions_now.data.cpu().numpy())
        self.targets.append(targets_now.data.cpu().numpy())
        
    def flush(self):
        self.predictions = np.concatenate(self.predictions, axis = 0)
        self.targets = np.concatenate(self.targets, axis = 0)
        
        output = {}
        output['rmse'] = get_rmse(self.predictions, self.targets)
        output['mae'] = get_mae(self.predictions, self.targets)
        output['relative rmse'] = get_relative_rmse(self.predictions, self.targets)
        
        self.predictions = []
        self.targets = []
        return output
    
class FullLogger:
    def __init__(self):
        self.train_logger = Logger()
        self.val_logger = Logger()        
        
    def flush(self):
        return {'train' : self.train_logger.flush(),
                'val' : self.val_logger.flush()}
    
    
def get_rotations(indices, global_aug = False): 
    if global_aug:
        num = np.max(indices) + 1
    else:
        num = indices.shape[0]
    
    rotations = Rotation.random(num).as_matrix()
    rotations[np.random.randn(rotations.shape[0]) >= 0] *= -1
    
    if global_aug:
        return rotations[indices]
    else:
        return rotations

def get_loss(predictions, targets):
    delta = predictions - targets
    return torch.mean(delta * delta)

def get_rmse(first, second):
    delta = first - second
    return np.sqrt(np.mean(delta * delta))

def get_mae(first, second):
    delta = first - second
    return np.mean(np.abs(delta))

def get_relative_rmse(predictions, targets):
    rmse = get_rmse(predictions, targets)
    return rmse / get_rmse(np.mean(targets), targets)
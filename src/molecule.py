
import torch
import ase.io
import numpy as np
from torch_geometric.data import Data


class Molecule():
    def __init__(self, atoms, r_cut, use_additional_scalar_attributes, use_forces, forces_key):
        
        self.use_additional_scalar_attributes = use_additional_scalar_attributes
        self.use_forces = use_forces
        
        self.atoms = atoms
        self.forces_key = forces_key

        positions = self.atoms.get_positions()
        species = self.atoms.get_atomic_numbers()
        
        self.central_species = []
        for i in range(len(positions)):
            self.central_species.append(species[i])
            
            
        if use_additional_scalar_attributes:
            scalar_attributes = self.atoms.arrays['scalar_attributes']
            if len(scalar_attributes.shape) == 1:
                scalar_attributes = scalar_attributes[:, np.newaxis]
        
            self.central_scalar_attributes = scalar_attributes
            

        
        i_list, j_list, D_list, S_list = ase.neighborlist.neighbor_list('ijDS', atoms, r_cut)
        
       
        self.neighbors_index = [[] for i in range(len(positions))]
        self.neighbors_shift = [[] for i in range(len(positions))]
        
        for i, j, D, S in zip(i_list, j_list, D_list, S_list):
            self.neighbors_index[i].append(j)
            self.neighbors_shift[i].append(S)
        
        
        
        self.relative_positions = [[] for i in range(len(positions))]
        self.neighbor_species = [[] for i in range(len(positions))]
        self.neighbors_pos = [[] for i in range(len(positions))]
        
        if use_additional_scalar_attributes:
            self.neighbor_scalar_attributes = [[] for i in range(len(positions))]
        
        def is_same(first, second):
            for i in range(len(first)):
                if first[i] != second[i]:
                    return False
            return True
            
        for i, j, D, S in zip(i_list, j_list, D_list, S_list):
            self.relative_positions[i].append(D)
            self.neighbor_species[i].append(species[j])
            if use_additional_scalar_attributes:
                self.neighbor_scalar_attributes[i].append(scalar_attributes[j])
            for k in range(len(self.neighbors_index[j])):
                if (self.neighbors_index[j][k] == i) and is_same(self.neighbors_shift[j][k], -S):
                    self.neighbors_pos[i].append(k)
                    
              
    def get_max_num(self):
        maximum = None
        for chunk in self.relative_positions:
            if (maximum is None) or (len(chunk) > maximum):
                maximum = len(chunk)
        return maximum
    
    def get_graph(self, max_num, all_species):
        central_species = [np.where(all_species == specie)[0][0] for specie in self.central_species]
        central_species = torch.LongTensor(central_species)
       
        
        nums = []
        mask = []
        relative_positions = np.zeros([len(self.relative_positions), max_num, 3])
        neighbors_pos = np.zeros([len(self.relative_positions), max_num], dtype = int)
        neighbors_index = np.zeros([len(self.relative_positions), max_num], dtype = int)
        
        if self.use_additional_scalar_attributes:
            neighbor_scalar_attributes = np.zeros([len(self.relative_positions), max_num, 1])
        
        for i in range(len(self.relative_positions)):
            now = np.array(self.relative_positions[i])
            if len(now) > 0:
                if self.use_additional_scalar_attributes:
                    neighbor_scalar_attributes[i, :len(now)] = self.neighbor_scalar_attributes[i]
                relative_positions[i, :len(now), :] = now
                neighbors_pos[i, :len(now)] = self.neighbors_pos[i]
                neighbors_index[i, :len(now)] = self.neighbors_index[i]
            
            nums.append(len(self.relative_positions[i]))
            current_mask = np.zeros(max_num)
            current_mask[len(self.relative_positions[i]):] = True
            mask.append(current_mask[np.newaxis, :])
            
            
        mask = np.concatenate(mask, axis = 0)
        relative_positions = torch.FloatTensor(relative_positions)
        nums = torch.FloatTensor(nums)
        mask = torch.BoolTensor(mask)
        
        neighbors_pos = torch.LongTensor(neighbors_pos)
        neighbors_index = torch.LongTensor(neighbors_index)
        
        neighbor_species = len(all_species) * np.ones([len(self.neighbor_species), max_num], dtype = int)
        for i in range(len(self.neighbor_species)):
            now = np.array(self.neighbor_species[i])
            now = np.array([np.where(all_species == specie)[0][0] for specie in now])
            neighbor_species[i, :len(now)] = now
        neighbor_species = torch.LongTensor(neighbor_species)
        
        kwargs = {'central_species' : central_species,
                  'x' : relative_positions,
                  'neighbor_species' : neighbor_species,
                  'neighbors_pos' : neighbors_pos,
                  'neighbors_index' : neighbors_index.transpose(0, 1),
                  'nums' : nums,
                  'mask' : mask}
        
        if self.use_forces:
            kwargs['forces'] = torch.FloatTensor(self.atoms.arrays[self.forces_key])
        
        if self.use_additional_scalar_attributes:
            kwargs['neighbor_scalar_attributes'] = torch.FloatTensor(neighbor_scalar_attributes)
            kwargs['central_scalar_attributes'] = torch.FloatTensor(self.central_scalar_attributes)
        
        result = Data(**kwargs)
    
        return result
    
def batch_to_dict(batch):
    batch_dict = {"x" : batch.x, 
                  "central_species" : batch.central_species,
                  "neighbor_species" : batch.neighbor_species,
                  "mask" : batch.mask,
                  "batch" : batch.batch, 
                  "nums" : batch.nums,
                  "neighbors_index" : batch.neighbors_index.transpose(0, 1),
                  "neighbors_pos" : batch.neighbors_pos}
    
    if hasattr(batch, 'neighbor_scalar_attributes'):
        batch_dict['neighbor_scalar_attributes'] = batch.neighbor_scalar_attributes
    if hasattr(batch, 'central_scalar_attributes'):
        batch_dict['central_scalar_attributes'] = batch.central_scalar_attributes
        
    return batch_dict
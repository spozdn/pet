import numpy as np
import math
import torch
from torch import nn

def get_upper_bound(vec, first_other, second_other, k_cut):
    normal = np.cross(first_other, second_other)
    normal = normal / np.sqrt(np.sum(normal * normal))
    bound = math.ceil(k_cut / np.abs(np.sum(vec * normal)))
    return bound


def get_all_k_from_reciprocal(w_1, w_2, w_3, k_cut):
    bound_1 = get_upper_bound(w_1, w_2, w_3, k_cut)
    bound_2 = get_upper_bound(w_2, w_1, w_3, k_cut)
    bound_3 = get_upper_bound(w_3, w_1, w_2, k_cut)

    result = []
    for first_index in range(-bound_1, bound_1 + 1):
        for second_index in range(-bound_2, bound_2 + 1):
            for third_index in range(-bound_3, bound_3 + 1):
                k_now = w_1 * first_index + w_2 * second_index + w_3 * third_index
                length_now = np.sqrt(np.sum(k_now * k_now))
                if length_now <= k_cut:
                    result.append(k_now)

    return result


def get_reciprocal(v_1, v_2, v_3):
    cross = np.cross(v_2, v_3)
    volume = np.abs(np.sum(v_1 * cross))
    w_1 = 2 * np.pi * np.cross(v_2, v_3) / volume
    w_2 = 2 * np.pi * np.cross(v_3, v_1) / volume
    w_3 = 2 * np.pi * np.cross(v_1, v_2) / volume
    return w_1, w_2, w_3


def get_all_k(v_1, v_2, v_3, k_cut):
    w_1, w_2, w_3 = get_reciprocal(v_1, v_2, v_3)
    return get_all_k_from_reciprocal(w_1, w_2, w_3, k_cut)


class LongRangeInteraction(torch.nn.Module):
    def __init__(self, hypers):
        super(LongRangeInteraction, self).__init__()
        self.hypers = hypers
        d_pet = hypers.TRANSFORMER_D_MODEL
        
        self.filter_calculator = nn.Sequential(nn.Linear(3, d_pet), get_activation(hypers),
                                    nn.Linear(d_pet, d_pet), get_activation(hypers),
                                    nn.Linear(d_pet, d_pet))
        
    def forward(self, k_vectors, positions, batch, h):
        s = get_s(k_vectors, positions, h, batch)
        filter_values = self.filter_calculator(k_vectors)
        predictions = get_new_h(k_vectors, positions, s, filter_values, batch)
        return predictions
        

def get_s(k_vectors, positions, h, batch):
    batch_size = k_vectors.shape[0]
    N_k = k_vectors.shape[1]
    d_pet = h.shape[1]
    
    k_vectors = k_vectors[batch]
    positions = positions[:, None, :]
    positions = positions.repeat(1, N_k, 1)
    
    k_pos = torch.sum(positions * k_vectors, dim = 2)
    h = h[:, None, :]
    h = h.repeat(1, N_k, 1)
    products = torch.exp(-1j * k_pos)[:, :, None] * h
    s = torch.zeros(batch_size, N_k, d_pet)
    s.index_add(0, batch, products)
    return s


def get_new_h(k_vectors, positions, s, filter_values, batch):
    N_k = k_vectors.shape[1]
    
    positions = positions.repeat(1, N_k, 1)
    k_pos = torch.sum(positions * k_vectors, dim = 2)
    k_pos = torch.exp(1j * k_pos)
    
    s = s[batch]
    filter_values = filter_values[batch]
    filter_values = filter_values[:, None, :].repeat(1, N_k, 1)
    
    k_pos = k_pos[:, None, :].repeat(1, N_k, 1)
    product = k_pos * s * filter_values
    return torch.sum(product, dim = 1)

from scipy.special import lambertw
import torch
import numpy as np

def smooth_max_weighted(values, weights, beta):
   
    numenator = 0.0
    denumenator = 0.0
    
    for value, weight in zip(values, weights):
        
        denumenator += torch.exp(value * beta) * weight
        numenator += torch.exp(value * beta) * weight * value
        
    
    return numenator / denumenator

def smooth_max(values, beta):
    weights = [1.0 for value in values]
    return smooth_max_weighted(values, weights, beta)


def get_q_func(grid, w, delta):
    mask_smaller = grid <= w
    mask_bigger = grid >= w + delta
    grid = (grid - w) / delta
    f = 0.5 - torch.cos(np.pi * grid) / 2.0
    f[mask_bigger] = 1.0
    f[mask_smaller] = 0.0
    return f

def cutoff_func(grid, r_cut, delta):
    
    mask_bigger = grid >= r_cut
    mask_smaller = grid <= r_cut - delta
    grid = (grid - r_cut + delta) / delta
    f = 1/2.0 + torch.cos(np.pi * grid)/2.0
    
    f[mask_bigger] = 0.0
    f[mask_smaller] = 1.0
    return f


def get_length(vec):
    return torch.sqrt(torch.sum(vec ** 2))


def get_normalized(vec):
        return vec / get_length(vec)
    
    
    
def get_coor_system(first_vec, second_vec):
        first_vec_normalized = get_normalized(first_vec)
        second_vec_normalized = get_normalized(second_vec)

        vec_product = torch.cross(first_vec_normalized, second_vec_normalized)
        vec_product_normalized = get_normalized(vec_product)

        last_one = torch.cross(first_vec_normalized, vec_product_normalized)

        coor_system = [first_vec_normalized[None, :],
                       vec_product_normalized[None, :],
                       last_one[None, :]]
        coor_system = torch.cat(coor_system, axis = 0)
        return torch.transpose(coor_system, -1, -2)

class SPHypers():
    def __init__(self, beta, delta, w):
        self.beta = beta
        self.delta = delta
        self.w = w        
        

class SPFramesCalculator():
    def __init__(self, sp_hypers):
        self.sp_hypers = sp_hypers
        self.lambert_constant = torch.tensor(float(lambertw(np.exp(-1.0))))
    
    def T_func(self, beta):
        return self.lambert_constant / beta
    
    def get_r_cut_inner(self, env, r_cut_outer):
        values, weights = [r_cut_outer], [1.0]

        for first_index in range(len(env)):
            for second_index in range(first_index + 1, len(env)):
                first_vector = env[first_index]
                second_vector = env[second_index]

                first_length = get_length(first_vector)
                second_length = get_length(second_vector)

                value_now = smooth_max([first_length, second_length], self.sp_hypers.beta)
                #print(value_now, self.T_func(self.sp_hypers.beta))
                value_now = value_now + self.T_func(self.sp_hypers.beta)

                values.append(value_now)

                first_weight_now = cutoff_func(first_length[None], r_cut_outer, self.sp_hypers.delta)[0]
                second_weight_now = cutoff_func(second_length[None], r_cut_outer, self.sp_hypers.delta)[0]

                first_normalized = get_normalized(first_vector)
                second_normalized = get_normalized(second_vector)

                vec_product = torch.cross(first_normalized, second_normalized)
                third_weight_now = get_q_func(torch.sum(vec_product ** 2)[None], self.sp_hypers.w, self.sp_hypers.delta)[0]

                weight_now = first_weight_now * second_weight_now * third_weight_now
                weights.append(weight_now)
        
        return smooth_max_weighted(values, weights, -self.sp_hypers.beta)
    
    
    def get_all_frames(self, env, r_cut_outer):
        r_cut_inner = self.get_r_cut_inner(env, r_cut_outer)
        coor_systems, weights = [], []
        for first_index in range(len(env)):
            for second_index in range(len(env)):
                if first_index != second_index:
                    first_vec = env[first_index]
                    second_vec = env[second_index]
       
                    first_length = get_length(first_vec)
                    second_length = get_length(second_vec)

                    if (first_length < r_cut_inner) and (second_length < r_cut_inner):
                        first_normalized = get_normalized(first_vec)
                        second_normalized = get_normalized(second_vec)

                        vec_product = torch.cross(first_normalized, second_normalized)
                        spread = torch.sum(vec_product ** 2)
                        if spread > self.sp_hypers.w:
                            coor_system = get_coor_system(first_vec, second_vec)

                            first_weight = cutoff_func(first_length[None], r_cut_inner, self.sp_hypers.delta)[0]
                            second_weight = cutoff_func(second_length[None], r_cut_inner, self.sp_hypers.delta)[0]
                            third_weight = get_q_func(spread, self.sp_hypers.w, self.sp_hypers.delta)

                            weight = first_weight * second_weight * third_weight

                            coor_systems.append(coor_system)
                            weights.append(weight)

        return coor_systems, weights
    
    
    
    def get_all_frames_global(self, envs_list, r_cut_outer):
        coor_systems, weights = [], []
        for env in envs_list:
            coor_systems_now, weights_now = self.get_all_frames(env, r_cut_outer)

            for el in coor_systems_now:
                coor_systems.append(el)

            for el in weights_now:
                weights.append(el)
        return coor_systems, weights
    
    
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

                value_now = smooth_max([first_length, second_length], self.sp_hypers.BETA)
                #print(value_now, self.T_func(self.sp_hypers.beta))
                value_now = value_now + self.T_func(self.sp_hypers.BETA)

                values.append(value_now)

                first_weight_now = cutoff_func(first_length[None], r_cut_outer, self.sp_hypers.DELTA)[0]
                second_weight_now = cutoff_func(second_length[None], r_cut_outer, self.sp_hypers.DELTA)[0]

                first_normalized = get_normalized(first_vector)
                second_normalized = get_normalized(second_vector)

                vec_product = torch.cross(first_normalized, second_normalized)
                third_weight_now = get_q_func(torch.sum(vec_product ** 2)[None], self.sp_hypers.W, self.sp_hypers.DELTA)[0]

                weight_now = first_weight_now * second_weight_now * third_weight_now
                weights.append(weight_now)
        
        return smooth_max_weighted(values, weights, -self.sp_hypers.BETA) #smooth_min with -beta
    
    
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
                        if spread > self.sp_hypers.W:
                            coor_system = get_coor_system(first_vec, second_vec)

                            first_weight = cutoff_func(first_length[None], r_cut_inner, self.sp_hypers.DELTA)[0]
                            second_weight = cutoff_func(second_length[None], r_cut_inner, self.sp_hypers.DELTA)[0]
                            third_weight = get_q_func(spread, self.sp_hypers.W, self.sp_hypers.DELTA)

                            weight = first_weight * second_weight * third_weight

                            coor_systems.append(coor_system)
                            weights.append(weight)

        return coor_systems, weights
    
    
    
    def get_all_frames_global(self, envs_list, r_cut_outer, epsilon = 1e-10):
        coor_systems, weights = [], []
        for env in envs_list:
            coor_systems_now, weights_now = self.get_all_frames(env, r_cut_outer)

            for el in coor_systems_now:
                coor_systems.append(el)

            for el in weights_now:
                weights.append(el)
        
        if len(weights) == 0:
            return [], [], torch.tensor(1.0, dtype = torch.float32).to(envs_list[0].device)
        
        
        max_weight = smooth_max(weights, self.sp_hypers.BETA)
        factors = get_q_func(torch.cat([weight[None] for weight in weights]), max_weight * self.sp_hypers.PRUNNING_THRESHOLD, max_weight * self.sp_hypers.PRUNNING_THRESHOLD_DELTA)
        #print(factors)
        #print(max_weight)
        coor_systems_final, weights_final = [], []
        for i in range(len(weights)):
            now = weights[i] * factors[i]
            if now > epsilon:
                weights_final.append(now)
                coor_systems_final.append(coor_systems[i])
        
        weight_aux = cutoff_func(max_weight[None], self.sp_hypers.AUX_THRESHOLD, self.sp_hypers.AUX_THRESHOLD_DELTA)[0]
        #print(type(weight_aux))
        return coor_systems_final, weights_final, weight_aux
    
    
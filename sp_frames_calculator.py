from scipy.special import lambertw
import torch
import numpy as np
import sys


def smooth_max_weighted(values, weights, beta):
   
    numenator = 0.0
    denumenator = 0.0
    
    for value, weight in zip(values, weights):
        #print(value, beta, weight)
        denumenator += torch.exp(value * beta) * weight
        numenator += torch.exp(value * beta) * weight * value
        
    #print(denumenator)
    return numenator / denumenator

def smooth_max(values, beta):
    weights = [1.0 for value in values]
    return smooth_max_weighted(values, weights, beta)


def q_func_exp(grid, w, delta, add_linear_multiplier):
    values = torch.zeros_like(grid)
    
    mask_bigger = grid > w
    values[mask_bigger] = torch.exp(-1.0 / ((grid[mask_bigger] - w) / delta))
    
    mask_smaller = grid <= w
    values[mask_smaller] = 0.0
    
    if add_linear_multiplier:
        mask_active = grid >= w
        values[mask_active] *= (grid[mask_active] - w) / delta
    return values

def cutoff_func_exp(grid, r_cut, delta):
    values = torch.zeros_like(grid)
    
    mask_smaller = grid < r_cut
    values[mask_smaller] = torch.exp(-1.0 / ((r_cut - grid[mask_smaller]) / delta))
    
    mask_bigger = grid >= r_cut
    values[mask_bigger] = 0.0
    return values

def get_normalized_tanh_stitching(x):
    return torch.tanh(1 / (x + 1) + 1/(x - 1))

def cutoff_func_tanh(grid, r_cut, delta):
    grid_normalized = 2.0 * (grid - (r_cut - 0.5 * delta)) / delta
    mask_smaller = grid_normalized <= -1.0
    mask_bigger = grid_normalized >= 1.0
    mask_switch = torch.logical_and(grid_normalized > -1.0, grid_normalized < 1.0)
    
    values = torch.zeros_like(grid)
    values[mask_smaller] = 1.0
    values[mask_bigger] = 0.0
    
    
    values[mask_switch] = 0.5 * get_normalized_tanh_stitching(grid_normalized[mask_switch]) + 0.5
    return values


def q_func_tanh(grid, w, delta, add_linear_multiplier):
    grid_normalized =  2.0 * (grid - (w + 0.5 * delta)) / delta
    
    
    mask_smaller = grid_normalized <= -1.0
    mask_bigger = grid_normalized >= 1.0
    mask_switch = torch.logical_and(grid_normalized > -1.0, grid_normalized < 1.0)
    
    values = torch.zeros_like(grid)
    values[mask_smaller] = 0.0
    values[mask_bigger] = 1.0

    values[mask_switch] = 0.5 * get_normalized_tanh_stitching(-grid_normalized[mask_switch]) + 0.5
    
    if add_linear_multiplier:
        mask_active = grid >= w
        values[mask_active] *= (grid[mask_active] - w)
    return values

def q_func(grid, w, delta, q_func_mode, add_linear_multiplier):
    if q_func_mode == 'exp':
        return q_func_exp(grid, w, delta, add_linear_multiplier)
    if q_func_mode == 'tanh':
        return q_func_tanh(grid, w, delta, add_linear_multiplier)
    raise ValueError("unknown mode for q func")
    
def cutoff_func(grid, r_cut, delta, cutoff_func_mode):
    if cutoff_func_mode == 'exp':
        return cutoff_func_exp(grid, r_cut, delta)
    if cutoff_func_mode == 'tanh':
        return cutoff_func_tanh(grid, r_cut, delta)
    raise ValueError('unknown mode for cutoff func')

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
        values, weights = [torch.tensor(r_cut_outer).to(env.device)], [1.0]

        for first_index in range(len(env)):
            for second_index in range(first_index + 1, len(env)):
                first_vector = env[first_index]
                second_vector = env[second_index]

                first_length = get_length(first_vector)
                second_length = get_length(second_vector)
                if (first_length <= r_cut_outer) and (second_length <= r_cut_outer):
                    value_now = smooth_max([first_length, second_length], self.sp_hypers.BETA)
                    #print(value_now, self.T_func(self.sp_hypers.beta))
                    value_now = value_now + self.T_func(self.sp_hypers.BETA)

                    values.append(value_now)

                    first_weight_now = cutoff_func(first_length[None], r_cut_outer, self.sp_hypers.DELTA_R_CUT, self.sp_hypers.CUTOFF_FUNC_MODE)[0]
                    second_weight_now = cutoff_func(second_length[None], r_cut_outer, self.sp_hypers.DELTA_R_CUT, self.sp_hypers.CUTOFF_FUNC_MODE)[0]

                    first_normalized = get_normalized(first_vector)
                    second_normalized = get_normalized(second_vector)

                    vec_product = torch.cross(first_normalized, second_normalized)
                    third_weight_now = q_func(torch.sum(vec_product ** 2)[None], self.sp_hypers.W, self.sp_hypers.DELTA_W, self.sp_hypers.Q_FUNC_MODE, False)[0]

                    weight_now = first_weight_now * second_weight_now * third_weight_now
                    weights.append(weight_now)
        
        return smooth_max_weighted(values, weights, -self.sp_hypers.BETA) + self.sp_hypers.DELTA_R_CUT #smooth_min with -beta
    
    
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
                        if spread > (self.sp_hypers.W - self.sp_hypers.DELTA_W):
                            coor_system = get_coor_system(first_vec, second_vec)

                            first_weight = cutoff_func(first_length[None], r_cut_inner, self.sp_hypers.DELTA_R_CUT, self.sp_hypers.CUTOFF_FUNC_MODE)[0]
                            second_weight = cutoff_func(second_length[None], r_cut_inner, self.sp_hypers.DELTA_R_CUT, self.sp_hypers.CUTOFF_FUNC_MODE)[0]
                            third_weight = q_func(spread, (self.sp_hypers.W - self.sp_hypers.DELTA_W), self.sp_hypers.DELTA_W, self.sp_hypers.Q_FUNC_MODE, self.sp_hypers.ADD_LINEAR_MULTIPLIER_Q_FUNC)

                            weight = first_weight * second_weight * third_weight

                            coor_systems.append(coor_system)
                            weights.append(weight)

        return coor_systems, weights
    
    
    
    def get_all_frames_global(self, envs_list, r_cut_initial, epsilon = 1e-10):
        r_cut_outer = min(r_cut_initial, self.sp_hypers.R_CUT_OUTER_UPPER_BOUND)
        coor_systems, weights = [], []
        for env in envs_list:
            coor_systems_now, weights_now = self.get_all_frames(env, r_cut_outer)

            for el in coor_systems_now:
                coor_systems.append(el)

            for el in weights_now:
                weights.append(el)
        
        if len(weights) == 0:
            zero_torch = torch.tensor(0.0, dtype = torch.float32).to(envs_list[0].device)
            return [], [], cutoff_func(zero_torch[None], self.sp_hypers.AUX_THRESHOLD, self.sp_hypers.AUX_THRESHOLD_DELTA, self.sp_hypers.CUTOFF_FUNC_MODE)[0]
        
        
        for _ in range(self.sp_hypers.NUM_PRUNNINGS):
            weights = torch.cat([weight[None] for weight in weights])
            max_weight = smooth_max_weighted(weights, weights, self.sp_hypers.BETA_WEIGHTS)

            factors = q_func(weights, max_weight * self.sp_hypers.PRUNNING_THRESHOLD, max_weight * self.sp_hypers.PRUNNING_THRESHOLD_DELTA, self.sp_hypers.Q_FUNC_MODE, False)
            #print(factors)
            #print(max_weight)
            coor_systems_final, weights_final = [], []
            for i in range(len(weights)):
                now = weights[i] * factors[i]
                if now > epsilon:
                    weights_final.append(now)
                    coor_systems_final.append(coor_systems[i])
            weights = weights_final
            coor_systems = coor_systems_final
        
        weights = torch.cat([weight[None] for weight in weights])
        max_weight = smooth_max_weighted(weights, weights, self.sp_hypers.BETA_WEIGHTS)
        
        weight_aux = cutoff_func(max_weight[None], self.sp_hypers.AUX_THRESHOLD, self.sp_hypers.AUX_THRESHOLD_DELTA, self.sp_hypers.CUTOFF_FUNC_MODE)[0]
        
        #print(len(weights), len(weights_final), max_weight, torch.max(weights))
        #np.set_printoptions(threshold=sys.maxsize)
        #for_printing = [weight.data.cpu().numpy() for weight in weights_final]
        #print(np.sort(for_printing[:]) * 100)
        #print('in sp frame calculator: ', max_weight, weight_aux)
        return coor_systems, weights, weight_aux
    
    
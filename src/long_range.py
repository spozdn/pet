import numpy as np
import math


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

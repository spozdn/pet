import numpy as np
import math

def get_structural_batch_size(structures, atomic_batch_size):
    sizes = [len(structure.get_positions()) for structure in structures]
    average_size = np.mean(sizes)
    return math.ceil(atomic_batch_size / average_size)

def convert_atomic_throughput(train_structures, atomic_throughput):
    sizes = [len(structure.get_positions()) for structure in train_structures]
    total_size = np.sum(sizes)
    return math.ceil(atomic_throughput / total_size)



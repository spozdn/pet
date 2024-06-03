import ase.io
import numpy as np
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
import argparse
from matscipy.neighbours import neighbour_list as neighbor_list

def parse_r_cut(value):
    try:
        # Try to convert the input to a single float
        return float(value)
    except ValueError:
        # If it fails, try to parse it as a tuple of floats
        try:
            values = tuple(map(float, value.strip("()").split(",")))
            if len(values) != 3:
                raise ValueError
            return values
        except ValueError:
            raise argparse.ArgumentTypeError("r_cut must be a single float or a tuple of three floats (start, stop, num)")

description = "Simple utility to compute average number of neighbors over whole the provided dataset for single or several cutoff radiuses"
parser = argparse.ArgumentParser(description = description)
parser.add_argument("structures_path", help="Path to an xyz file with structures", type = str)
parser.add_argument("r_cut", help = "cutoff radius", type = parse_r_cut)
parser.add_argument("--num_strucs_use", help="Number of random structures to use to estimate statics. If not specified, all the provided structures are used", type = str)

args = parser.parse_args()

structures_path = args.structures_path

structures = ase.io.read(structures_path, index = ':')

if args.num_strucs_use is not None:
    permutation = np.random.permutation(len(structures))
    structures = [structures[index] for index in permutation[:args.num_strucs_use]]

atom_nums = [len(struc.positions) for struc in structures]
total_num = np.sum(atom_nums)

p = Pool(cpu_count())

def get_n_neighbors(atoms, r_cut):
    def is_3d_crystal(atoms):
        pbc = atoms.get_pbc()
        if isinstance(pbc, bool):
            return pbc
        return all(pbc)

    if is_3d_crystal(atoms):
        i_list, j_list = neighbor_list('ij', atoms, r_cut)
    else:
        i_list, j_list = ase.neighborlist.neighbor_list('ij', atoms, r_cut)
    return len(i_list)

if isinstance(args.r_cut, float):
    num_neighbors = p.map(get_n_neighbors, structures)
    total_neighbors = np.sum(num_neighbors)
    average_n_neighbors = total_neighbors / total_num
    print(f"average number of neighbors within r_cut = {args.r_cut} is {average_n_neighbors}")
    print("total number of atomic environment used to compute statistics is :", total_num)
else:
    start, stop, num = args.r_cut
    print("Getting statistics for several cutoffs from the provided range:")
    for r_cut in np.linspace(start, stop, num):
        num_neighbors = p.map(get_n_neighbors, structures)
        total_neighbors = np.sum(num_neighbors)
        average_n_neighbors = total_neighbors / total_num
        print(f"There are {average_n_neighbors} neighbors on average for r_cut = {r_cut}.")
    print("total number of atomic environment used to compute statistics is :", total_num)

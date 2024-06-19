import argparse
import numpy as np
from ase.io import read
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from matscipy.neighbours import neighbour_list as neighbor_list

def parse_r_cut(value):
    try:
        return float(value)
    except ValueError:
        try:
            values = tuple(map(float, value.strip("()").split(",")))
            if len(values) != 3:
                raise ValueError
            return values
        except ValueError:
            raise argparse.ArgumentTypeError("r_cut must be a single float or a tuple of three floats (start, stop, num)")

def get_n_neighbors(atoms, r_cut):
    from matscipy.neighbours import neighbour_list as m_neighbor_list
    from ase.neighborlist import neighbor_list
    
    def is_3d_crystal(atoms):
        pbc = atoms.get_pbc()
        return pbc if isinstance(pbc, bool) else all(pbc)

    if is_3d_crystal(atoms):
        i_list, j_list = m_neighbor_list('ij', atoms, r_cut)
    else:
        i_list, j_list = neighbor_list('ij', atoms, r_cut)
    return len(i_list)

def main():
    description = ("Simple utility to compute average number of neighbors "
                   "over the provided dataset for single or several cutoff radii")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("structures_path", help="Path to an xyz file with structures", type=str)
    parser.add_argument("r_cut", help=("Cutoff radius. Can be provided as a single float or "
                                       "as a tuple in the format (start, stop, num), where "
                                       "start and stop are floats and num is an integer. "
                                       "If a tuple is provided, the script will use all cutoff values "
                                       "from the linspace defined by these values. "
                                       "For example for (1.0, 2.0, 5) statistics will be estimated"
                                       "for all cutoff radii from [1.0, 1.25, 1.5, 1.75, 2.0]"), type=parse_r_cut)
    parser.add_argument("--num_strucs_use", help=("Number of random structures to use to estimate "
                                                  "statistics. If not specified, all the provided structures are used"), type=int)

    args = parser.parse_args()

    structures_path = args.structures_path
    structures = read(structures_path, index=':')

    if args.num_strucs_use is not None:
        permutation = np.random.permutation(len(structures))
        structures = [structures[index] for index in permutation[:args.num_strucs_use]]

    atom_nums = [len(struc.positions) for struc in structures]
    total_num = np.sum(atom_nums)

    p = Pool(cpu_count())

    if isinstance(args.r_cut, float):
        num_neighbors = p.map(lambda atoms: get_n_neighbors(atoms, args.r_cut), structures)
        total_neighbors = np.sum(num_neighbors)
        average_n_neighbors = total_neighbors / total_num
        print(f"Average number of neighbors within r_cut = {args.r_cut:.3f} is {average_n_neighbors:.3f}")
        print("Total number of atomic environments used to compute statistics is:", total_num)
    else:
        start, stop, num = args.r_cut
        print("Getting statistics for several cutoffs from the provided range:")
        for r_cut in np.linspace(start, stop, int(num)):
            num_neighbors = p.map(lambda atoms: get_n_neighbors(atoms, r_cut), structures)
            total_neighbors = np.sum(num_neighbors)
            average_n_neighbors = total_neighbors / total_num
            print(f"There are {average_n_neighbors:.3f} neighbors on average for r_cut = {r_cut:.3f}.")
        print("Total number of atomic environments used to compute statistics is:", total_num)

if __name__ == "__main__":
    main()

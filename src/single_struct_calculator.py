
import torch
import numpy as np
from torch_geometric.nn import DataParallel

from .data_preparation import get_compositional_features
from .molecule import Molecule
from .hypers import Hypers
from .pet import PET, PETMLIPWrapper


class SingleStructCalculator():
    def __init__(self, path_to_calc_folder, checkpoint="best_val_rmse_both_model", default_hypers_path="default_hypers.yaml", device="cpu"): 
        hypers_path = path_to_calc_folder + '/hypers_used.yaml'
        path_to_model_state_dict = path_to_calc_folder + '/' + checkpoint + '_state_dict'
        all_species_path = path_to_calc_folder + '/all_species.npy'
        self_contributions_path = path_to_calc_folder + '/self_contributions.npy'
        
        hypers = Hypers()
        hypers.set_from_files(hypers_path, default_hypers_path, check_duplicated = False)
        
        all_species = np.load(all_species_path)
        if hypers.USE_ENERGIES:
            self.self_contributions = np.load(self_contributions_path)
            
        add_tokens = []
        for _ in range(hypers.N_GNN_LAYERS - 1):
            add_tokens.append(hypers.ADD_TOKEN_FIRST)
        add_tokens.append(hypers.ADD_TOKEN_SECOND)

        model = PET(hypers, hypers.TRANSFORMER_D_MODEL, hypers.TRANSFORMER_N_HEAD,
                               hypers.TRANSFORMER_DIM_FEEDFORWARD, hypers.N_TRANS_LAYERS, 
                               0.0, len(all_species), 
                               hypers.N_GNN_LAYERS, hypers.HEAD_N_NEURONS, hypers.TRANSFORMERS_CENTRAL_SPECIFIC, hypers.HEADS_CENTRAL_SPECIFIC, 
                               add_tokens).to(device)
        model = PETMLIPWrapper(model, hypers)
        if hypers.MULTI_GPU and torch.cuda.is_available():
            model = DataParallel(model)
            model = model.to( torch.device('cuda:0'))

        model.load_state_dict(torch.load(path_to_model_state_dict, map_location=torch.device(device)))
        model.eval()
        
        self.model = model
        self.hypers = hypers
        self.all_species = all_species
        
        
    def forward(self, structure):
        molecule = Molecule(structure, self.hypers.R_CUT, 
                            self.hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES)
        
        graph = molecule.get_graph(molecule.get_max_num(), self.all_species)
        graph.y = 0
        graph.forces = np.zeros_like(structure.positions)
        prediction_energy, _, prediction_forces, _ = self.model(graph, augmentation = False, create_graph = False)

        compositional_features = get_compositional_features([structure], self.all_species)[0]
        self_contributions_energy = np.dot(compositional_features, self.self_contributions)
        energy_total = prediction_energy.data.cpu().numpy() + self_contributions_energy
        return energy_total, prediction_forces.data.cpu().numpy()


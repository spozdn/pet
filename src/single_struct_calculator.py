
import torch
import numpy as np
from torch_geometric.nn import DataParallel

from .data_preparation import get_compositional_features
from .molecule import Molecule
from .hypers import load_hypers_from_file
from .pet import PET, PETMLIPWrapper


class SingleStructCalculator():
    def __init__(self, path_to_calc_folder, checkpoint="best_val_rmse_both_model", device="cpu"): 
        hypers_path = path_to_calc_folder + '/hypers_used.yaml'
        path_to_model_state_dict = path_to_calc_folder + '/' + checkpoint + '_state_dict'
        all_species_path = path_to_calc_folder + '/all_species.npy'
        self_contributions_path = path_to_calc_folder + '/self_contributions.npy'
        
        hypers = load_hypers_from_file(hypers_path)
        
        MLIP_SETTINGS = hypers.MLIP_SETTINGS
        ARCHITECTURAL_HYPERS = hypers.ARCHITECTURAL_HYPERS
        FITTING_SCHEME = hypers.FITTING_SCHEME

        self.architectural_hypers = ARCHITECTURAL_HYPERS

        all_species = np.load(all_species_path)
        if MLIP_SETTINGS.USE_ENERGIES:
            self.self_contributions = np.load(self_contributions_path)
            
        model = PET(ARCHITECTURAL_HYPERS, 0.0, len(all_species),
                FITTING_SCHEME.GLOBAL_AUG).to(device)
        
        model = PETMLIPWrapper(model, MLIP_SETTINGS.USE_ENERGIES, MLIP_SETTINGS.USE_FORCES)
        if FITTING_SCHEME.MULTI_GPU and torch.cuda.is_available():
            model = DataParallel(model)
            model = model.to(torch.device('cuda:0'))

        model.load_state_dict(torch.load(path_to_model_state_dict, map_location=torch.device(device)))
        model.eval()
        
        self.model = model
        self.hypers = hypers
        self.all_species = all_species
        
        
    def forward(self, structure):
        molecule = Molecule(structure, self.architectural_hypers.R_CUT, 
                            self.architectural_hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
                            self.architectural_hypers.USE_LONG_RANGE, self.architectural_hypers.K_CUT)
        
        graph = molecule.get_graph(molecule.get_max_num(), self.all_species, molecule.get_num_k())
       
        prediction_energy, prediction_forces = self.model(graph, augmentation = False, create_graph = False)

        compositional_features = get_compositional_features([structure], self.all_species)[0]
        self_contributions_energy = np.dot(compositional_features, self.self_contributions)
        energy_total = prediction_energy.data.cpu().numpy() + self_contributions_energy
        return energy_total, prediction_forces.data.cpu().numpy()


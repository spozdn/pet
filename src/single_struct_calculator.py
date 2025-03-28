import torch
import numpy as np
from torch_geometric.nn import DataParallel

from .data_preparation import get_compositional_features
from .molecule import Molecule, MoleculeCPP
from .hypers import load_hypers_from_file
from .pet import PET, PETMLIPWrapper, PETUtilityWrapper
from .utilities import string2dtype, get_quadrature_predictions


class SingleStructCalculator:
    def __init__(
        self, path_to_calc_folder, checkpoint="best_val_rmse_both_model", device="cpu", quadrature_order=None, inversions=False, use_augmentation=False, add_self_contributions=False,
    ):
        if (quadrature_order is not None) and (use_augmentation):
            raise NotImplementedError("Simultaneous use of a quadrature and augmentation is not yet implemented")

        self.use_augmentation = use_augmentation
        hypers_path = path_to_calc_folder + "/hypers_used.yaml"
        path_to_model_state_dict = (
            path_to_calc_folder + "/" + checkpoint + "_state_dict"
        )
        all_species_path = path_to_calc_folder + "/all_species.npy"
        self_contributions_path = path_to_calc_folder + "/self_contributions.npy"

        hypers = load_hypers_from_file(hypers_path)

        MLIP_SETTINGS = hypers.MLIP_SETTINGS
        ARCHITECTURAL_HYPERS = hypers.ARCHITECTURAL_HYPERS
        FITTING_SCHEME = hypers.FITTING_SCHEME

        self.architectural_hypers = ARCHITECTURAL_HYPERS

        all_species = np.load(all_species_path)
        if MLIP_SETTINGS.USE_ENERGIES:
            self.self_contributions = np.load(self_contributions_path)

        model = PET(ARCHITECTURAL_HYPERS, 0.0, len(all_species)).to(device)
        model = PETUtilityWrapper(model, FITTING_SCHEME.GLOBAL_AUG)

        model = PETMLIPWrapper(
            model, MLIP_SETTINGS.USE_ENERGIES, MLIP_SETTINGS.USE_FORCES
        )
        if torch.cuda.is_available() and (torch.cuda.device_count() > 1):
            model = DataParallel(model)
            model = model.to(torch.device("cuda:0"))

        model.load_state_dict(
            torch.load(path_to_model_state_dict,
                       map_location=torch.device(device))
        )

        if FITTING_SCHEME.MULTI_GPU and torch.cuda.is_available():
            model = model.module

        model.eval()

        self.model = model
        self.hypers = hypers
        self.all_species = all_species
        self.device = device

        if use_augmentation and (quadrature_order is not None or inversions):
            raise NotImplementedError("Simultaneous use of a quadrature/inversions and augmentation is not yet implemented")

        if quadrature_order is not None:
            self.quadrature_order = int(quadrature_order)
        else:
            self.quadrature_order = None

        self.inversions = inversions
        self.use_augmentation = use_augmentation
        self.add_self_contributions = add_self_contributions

    def forward(self, structure):
        molecule = MoleculeCPP(
            structure,
            self.architectural_hypers.R_CUT,
            self.architectural_hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
            self.architectural_hypers.USE_LONG_RANGE,
            self.architectural_hypers.K_CUT,
            self.architectural_hypers.N_TARGETS > 1,
            self.architectural_hypers.TARGET_INDEX_KEY
        )
        if self.architectural_hypers.USE_LONG_RANGE:
            raise NotImplementedError(
                "Long range interactions are not supported in the SingleStructCalculator"
            )

        graph = molecule.get_graph(
            molecule.get_max_num(), self.all_species, None
        )
        graph.batch = torch.zeros(
            graph.num_nodes, dtype=torch.long, device=graph.x.device
        )
        graph = graph.to(self.device)

        if self.quadrature_order is None and not self.inversions:
            prediction_energy, prediction_forces = self.model(
                graph, augmentation=self.use_augmentation, create_graph=False
            )
            prediction_energy_final = prediction_energy.data.cpu().numpy()
            prediction_forces_final = prediction_forces.data.cpu().numpy()
        else:
            prediction_energy_final, prediction_forces_final = get_quadrature_predictions(
                graph, self.model, self.quadrature_order, self.inversions, string2dtype(self.architectural_hypers.DTYPE)
            )

        energy_total = prediction_energy_final

        if self.add_self_contributions:
            compositional_features = get_compositional_features(
                [structure], self.all_species
            )[0]
            self_contributions_energy = np.dot(
                compositional_features, self.self_contributions
            )
            # note: this may lead to numerical problems in less than double precision
            energy_total = energy_total + self_contributions_energy

        return energy_total, prediction_forces_final

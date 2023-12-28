import os
import random
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from scipy.spatial.transform import Rotation
from torch_geometric.loader import DataLoader, DataListLoader
import copy


def get_calc_names(all_completed_calcs, current_name):
    name_to_load = None
    name_of_calculation = current_name
    if name_of_calculation in all_completed_calcs:
        name_to_load = name_of_calculation
        for i in range(100000):
            name_now = name_of_calculation + f"_continuation_{i}"
            if name_now not in all_completed_calcs:
                name_to_save = name_now
                break
            name_to_load = name_now
        name_of_calculation = name_to_save
    return name_to_load, name_of_calculation


def set_reproducibility(random_seed, cuda_deterministic):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    if cuda_deterministic and torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_length(delta):
    return np.sqrt(np.sum(delta * delta))


class ModelKeeper:
    def __init__(self):
        self.best_model = None
        self.best_error = None
        self.best_epoch = None
        self.additional_info = None

    def update(self, model_now, error_now, epoch_now, additional_info=None):
        if (self.best_error is None) or (error_now < self.best_error):
            self.best_error = error_now
            self.best_model = copy.deepcopy(model_now)
            self.best_epoch = epoch_now
            self.additional_info = additional_info


class Logger:
    def __init__(self):
        self.predictions = []
        self.targets = []

    def update(self, predictions_now, targets_now):
        self.predictions.append(predictions_now.data.cpu().numpy())
        self.targets.append(targets_now.data.cpu().numpy())

    def flush(self):
        self.predictions = np.concatenate(self.predictions, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

        output = {}
        output["rmse"] = get_rmse(self.predictions, self.targets)
        output["mae"] = get_mae(self.predictions, self.targets)
        output["relative rmse"] = get_relative_rmse(self.predictions, self.targets)

        self.predictions = []
        self.targets = []
        return output


class FullLogger:
    def __init__(self):
        self.train_logger = Logger()
        self.val_logger = Logger()

    def flush(self):
        return {"train": self.train_logger.flush(), "val": self.val_logger.flush()}


def get_rotations(indices, global_aug=False):
    if global_aug:
        num = np.max(indices) + 1
    else:
        num = indices.shape[0]

    rotations = Rotation.random(num).as_matrix()
    rotations[np.random.randn(rotations.shape[0]) >= 0] *= -1

    if global_aug:
        return rotations[indices]
    else:
        return rotations


def get_loss(predictions, targets):
    delta = predictions - targets
    return torch.mean(delta * delta)


def get_rmse(first, second):
    delta = first - second
    return np.sqrt(np.mean(delta * delta))


def get_mae(first, second):
    delta = first - second
    return np.mean(np.abs(delta))


def get_relative_rmse(predictions, targets):
    rmse = get_rmse(predictions, targets)
    return rmse / get_rmse(np.mean(targets), targets)


def get_scheduler(optim, FITTING_SCHEME):
    def func_lr_scheduler(epoch):
        if epoch < FITTING_SCHEME.EPOCHS_WARMUP:
            return epoch / FITTING_SCHEME.EPOCHS_WARMUP
        delta = epoch - FITTING_SCHEME.EPOCHS_WARMUP
        num_blocks = delta // FITTING_SCHEME.SCHEDULER_STEP_SIZE 
        return 0.5 ** (num_blocks)

    scheduler = LambdaLR(optim, func_lr_scheduler)
    return scheduler


def load_checkpoint(model, optim, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optim_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

def get_data_loaders(train_graphs, val_graphs, FITTING_SCHEME):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(FITTING_SCHEME.RANDOM_SEED)

    if FITTING_SCHEME.MULTI_GPU:
        train_loader = DataListLoader(train_graphs, batch_size=FITTING_SCHEME.STRUCTURAL_BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
        val_loader = DataListLoader(val_graphs, batch_size = FITTING_SCHEME.STRUCTURAL_BATCH_SIZE, shuffle = False, worker_init_fn=seed_worker, generator=g)
    else:
        train_loader = DataLoader(train_graphs, batch_size=FITTING_SCHEME.STRUCTURAL_BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_graphs, batch_size = FITTING_SCHEME.STRUCTURAL_BATCH_SIZE, shuffle = False, worker_init_fn=seed_worker, generator=g)

    return train_loader, val_loader
import pytest
import torch
import numpy as np

from pet.utilities import get_shift_agnostic_mse, get_shift_agnostic_mae, Logger

class Float64DtypeContext:
    def __enter__(self):
        # Save the current default dtype
        self.original_dtype = torch.get_default_dtype()
        # Set the default dtype to float64
        torch.set_default_dtype(torch.float64)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the original default dtype
        torch.set_default_dtype(self.original_dtype)

def generate_data(batch_size, prediction_length, target_length, num_batches):
    np.random.seed(0)
    predictions_batches = []
    targets_batches = []
    for _ in range(num_batches):
        predictions = np.random.rand(batch_size, prediction_length).astype(np.float64)
        targets = np.random.rand(batch_size, target_length).astype(np.float64)
        predictions_batches.append(predictions)
        targets_batches.append(targets)
    return predictions_batches, targets_batches

def test_logger_shift_agnostic_loss():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    support_missing_values = False
    use_shift_agnostic_loss = True
    epsilon = 1e-10
    
    # Generate test data
    batch_size = 32
    prediction_length = 76
    target_length = 35
    num_batches = 17
    predictions_batches, targets_batches = generate_data(batch_size, prediction_length, target_length, num_batches)

    with Float64DtypeContext():
        # Initialize logger
        logger = Logger(support_missing_values, use_shift_agnostic_loss, device)
        
        # Update logger with batches
        for predictions, targets in zip(predictions_batches, targets_batches):
            predictions_tensor = torch.tensor(predictions, device=device)
            targets_tensor = torch.tensor(targets, device=device)
            logger.update(predictions_tensor, targets_tensor)
        
        # Flush logger and get results
        output = logger.flush()
        
        # Concatenate all batches
        concatenated_predictions = np.concatenate(predictions_batches, axis=0)
        concatenated_targets = np.concatenate(targets_batches, axis=0)
        
        # Compute shift-agnostic losses for concatenated data
        concatenated_predictions_tensor = torch.tensor(concatenated_predictions, device=device)
        concatenated_targets_tensor = torch.tensor(concatenated_targets, device=device)
        
        expected_mse = get_shift_agnostic_mse(concatenated_predictions_tensor, concatenated_targets_tensor).item()
        expected_rmse = np.sqrt(expected_mse)
        expected_mae = get_shift_agnostic_mae(concatenated_predictions_tensor, concatenated_targets_tensor).item()
        
        # Compare the results
        assert abs(output['rmse'] - expected_rmse) < epsilon, f"RMSE mismatch: {output['rmse']} != {expected_rmse}"
        assert abs(output['mae'] - expected_mae) < epsilon, f"MAE mismatch: {output['mae']} != {expected_mae}"
import pytest
from pet.utilities import get_shift_agnostic_loss
import torch


def get_mse_loss(targets, predictions):
    delta = targets - predictions
    return torch.mean(delta * delta)


def get_shift_agnostic_loss_naive_1d(targets, predictions):
    best = None
    for shift in range(0, targets.shape[0] - predictions.shape[0] + 1):
        now = get_mse_loss(targets[shift : shift + predictions.shape[0]], predictions)
        if (best is None) or (now < best):
            best = now
    return best


def get_shift_agnostic_loss_naive(predictions, targets):
    losses = []
    for index in range(targets.shape[0]):
        losses.append(
            get_shift_agnostic_loss_naive_1d(targets[index], predictions[index])[None]
        )
    losses = torch.cat(losses)
    print(losses.shape)
    return torch.mean(losses)


def test_shift_agnostic_loss():
    """Compare the shift agnostic loss to the naive implementation
    with python loops"""
    predictions = torch.randn(128, 300)
    targets = torch.randn(128, 500)

    shift_agnostic_loss = get_shift_agnostic_loss(predictions, targets)
    shift_agnostic_loss_naive = get_shift_agnostic_loss_naive(predictions, targets)

    discrepancy = torch.abs(shift_agnostic_loss - shift_agnostic_loss_naive)
    assert discrepancy < 1e-6, "shift agnostic loss is not correct"

    shift_agnostic_loss = get_shift_agnostic_loss(targets, predictions)
    discrepancy = torch.abs(shift_agnostic_loss - shift_agnostic_loss_naive)
    assert discrepancy < 1e-6, "shift agnostic loss is not correct"

    predictions = torch.randn(128, 300)
    targets = torch.randn(128, 300)

    shift_agnostic_loss = get_shift_agnostic_loss(predictions, targets)
    loss_naive = torch.mean((predictions - targets) ** 2)
    discrepancy = torch.abs(shift_agnostic_loss - loss_naive)
    assert discrepancy < 1e-6, "shift agnostic loss is not correct"

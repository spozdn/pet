import torch
from typing import Dict, List


def _ensure_supported(batch_dict: Dict[str, torch.Tensor]) -> None:
    unsupported_keys: List[str] = [
        "neighbor_scalar_attributes",
        "central_scalar_attributes",
        "k_vectors",
        "k_mask",
        "positions",
        "volume",
        "target_id",
    ]
    present_unsupported = [key for key in unsupported_keys if key in batch_dict]
    if len(present_unsupported) > 0:
        raise ValueError(
            "permute_batch_dict currently supports only the case without long-range, "
            "without additional scalar attributes, and single-target training. "
            f"Unsupported keys present: {present_unsupported}"
        )


def _validate_required_keys(batch_dict: Dict[str, torch.Tensor]) -> None:
    required_keys: List[str] = [
        "x",
        "central_species",
        "neighbor_species",
        "mask",
        "batch",
        "nums",
        "neighbors_index",
        "neighbors_pos",
    ]
    missing = [key for key in required_keys if key not in batch_dict]
    if len(missing) > 0:
        raise ValueError(f"batch_dict is missing required keys: {missing}")


def _validate_shapes_and_dtypes(batch_dict: Dict[str, torch.Tensor]) -> None:
    x = batch_dict["x"]
    neighbor_species = batch_dict["neighbor_species"]
    mask = batch_dict["mask"]
    neighbors_index = batch_dict["neighbors_index"]
    neighbors_pos = batch_dict["neighbors_pos"]

    if x.dim() < 2:
        raise ValueError("'x' must have shape [N, max_num, ...]")

    n_nodes = x.shape[0]
    max_num = x.shape[1]

    if neighbor_species.shape[0] != n_nodes or neighbor_species.shape[1] != max_num:
        raise ValueError("'neighbor_species' must have shape [N, max_num]")
    if mask.shape[0] != n_nodes or mask.shape[1] != max_num:
        raise ValueError("'mask' must have shape [N, max_num]")
    if neighbors_index.shape[0] != n_nodes or neighbors_index.shape[1] != max_num:
        raise ValueError("'neighbors_index' must have shape [N, max_num]")
    if neighbors_pos.shape[0] != n_nodes or neighbors_pos.shape[1] != max_num:
        raise ValueError("'neighbors_pos' must have shape [N, max_num]")

    if neighbors_index.dtype != torch.long:
        raise ValueError("'neighbors_index' must have dtype torch.long")
    if neighbors_pos.dtype != torch.long:
        raise ValueError("'neighbors_pos' must have dtype torch.long")
    if batch_dict["central_species"].dtype != torch.long:
        raise ValueError("'central_species' must have dtype torch.long")
    if batch_dict["batch"].dtype != torch.long:
        raise ValueError("'batch' must have dtype torch.long")


def _validate_permutation(perm: torch.Tensor, n_nodes: int, device: torch.device) -> torch.Tensor:
    if not isinstance(perm, torch.Tensor):
        raise ValueError("perm must be a torch.Tensor")
    if perm.dtype != torch.long:
        raise ValueError("perm must be a 1D LongTensor")
    if perm.dim() != 1:
        raise ValueError("perm must be a 1D LongTensor")
    if perm.numel() != n_nodes:
        raise ValueError(f"perm length ({perm.numel()}) must equal number of nodes ({n_nodes})")

    perm_device = perm.device
    if perm_device != device:
        perm = perm.to(device)

    sorted_perm, _ = torch.sort(perm)
    if not torch.equal(sorted_perm, torch.arange(n_nodes, device=device)):
        raise ValueError("perm is not a valid permutation of 0..N-1")

    return perm


def permute_batch_dict(batch_dict: Dict[str, torch.Tensor], perm: torch.LongTensor) -> Dict[str, torch.Tensor]:
    """
    Return a new batch_dict with nodes permuted by 'perm' (global across the batch).

    Assumptions (enforced via ValueError):
    - No long-range tensors present (k_vectors, k_mask, positions, volume)
    - No additional scalar attributes (neighbor/central scalar attributes)
    - Single-target (no target_id)

    The function reorders all per-node and per-edge tensors along the node axis and
    remaps 'neighbors_index' values using the inverse permutation so that neighbor
    indices still refer to correct nodes under the new ordering.
    """

    _validate_required_keys(batch_dict)
    _ensure_supported(batch_dict)
    _validate_shapes_and_dtypes(batch_dict)

    x = batch_dict["x"]
    device = x.device
    n_nodes = x.shape[0]

    perm = _validate_permutation(perm, n_nodes, device)

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(n_nodes, device=device)

    result: Dict[str, torch.Tensor] = dict(batch_dict)

    keys_to_permute_along_node_axis: List[str] = [
        "x",
        "neighbor_species",
        "mask",
        "neighbors_pos",
        "central_species",
        "batch",
        "nums",
    ]
    for key in keys_to_permute_along_node_axis:
        if key in batch_dict and batch_dict[key] is not None:
            result[key] = batch_dict[key].index_select(0, perm)

    if "input_messages" in batch_dict and batch_dict["input_messages"] is not None:
        result["input_messages"] = batch_dict["input_messages"].index_select(0, perm)

    neighbors_index = batch_dict["neighbors_index"].index_select(0, perm)
    neighbors_index = inv_perm[neighbors_index]
    result["neighbors_index"] = neighbors_index

    return result



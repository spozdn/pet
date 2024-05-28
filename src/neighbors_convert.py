import torch
from torch.autograd import Function

class ProcessNeighborsFunction(Function):
    @staticmethod
    def forward(ctx, i_list, j_list, S_list, D_list, max_size, n_atoms, species, scalar_attributes, all_species):
        outputs = torch.ops.neighbors_convert.process(
            i_list, j_list, S_list, D_list, max_size, n_atoms, species, scalar_attributes, all_species
        )
        ctx.save_for_backward(i_list)
        ctx.max_size = max_size
        ctx.n_atoms = n_atoms
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_relative_positions = grad_outputs[1]  # Assuming this is the gradient w.r.t relative_positions tensor
        i_list, = ctx.saved_tensors
        grad_D_list = torch.ops.neighbors_convert.process_backward(
            grad_relative_positions, i_list, ctx.max_size, ctx.n_atoms
        )
        # Return gradients for inputs
        return None, None, None, grad_D_list, None, None, None, None, None

# Utility function to use the custom autograd function
def process_neighbors(i_list, j_list, S_list, D_list, max_size, n_atoms, species, scalar_attributes, all_species):
    return ProcessNeighborsFunction.apply(i_list, j_list, S_list, D_list, max_size, n_atoms, species, scalar_attributes, all_species)
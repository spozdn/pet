#include <torch/extension.h>
#include <vector>
#include <algorithm>  // For std::fill
#include <c10/util/Optional.h>  // For c10::optional

// Template function to process the neighbors
template <typename int_t, typename float_t>
std::vector<c10::optional<at::Tensor>> process_neighbors(at::Tensor i_list, at::Tensor j_list, at::Tensor S_list, at::Tensor D_list, 
                                                        int64_t max_size, int64_t n_atoms, at::Tensor species,
                                                        c10::optional<at::Tensor> scalar_attributes,
                                                        at::Tensor all_species) {
    // Ensure the tensors are on the CPU and are contiguous
    TORCH_CHECK(i_list.device().is_cpu(), "i_list must be on CPU");
    TORCH_CHECK(j_list.device().is_cpu(), "j_list must be on CPU");
    TORCH_CHECK(S_list.device().is_cpu(), "S_list must be on CPU");
    TORCH_CHECK(D_list.device().is_cpu(), "D_list must be on CPU");
    TORCH_CHECK(species.device().is_cpu(), "species must be on CPU");
    TORCH_CHECK(all_species.device().is_cpu(), "all_species must be on CPU");

    TORCH_CHECK(i_list.is_contiguous(), "i_list must be contiguous");
    TORCH_CHECK(j_list.is_contiguous(), "j_list must be contiguous");
    TORCH_CHECK(S_list.is_contiguous(), "S_list must be contiguous");
    TORCH_CHECK(D_list.is_contiguous(), "D_list must be contiguous");
    TORCH_CHECK(species.is_contiguous(), "species must be contiguous");
    TORCH_CHECK(all_species.is_contiguous(), "all_species must be contiguous");

    // Ensure the sizes match
    TORCH_CHECK(i_list.sizes() == j_list.sizes(), "i_list and j_list must have the same size");
    TORCH_CHECK(i_list.size(0) == S_list.size(0) && S_list.size(1) == 3, "S_list must have the shape [N, 3]");
    TORCH_CHECK(i_list.size(0) == D_list.size(0) && D_list.sizes() == S_list.sizes(), "D_list must have the same shape as S_list");

    // Initialize tensors with zeros
    auto options_int = torch::TensorOptions().dtype(i_list.dtype()).device(torch::kCPU);
    auto options_float = torch::TensorOptions().dtype(D_list.dtype()).device(torch::kCPU);
    auto options_bool = torch::TensorOptions().dtype(at::kBool).device(torch::kCPU);

    at::Tensor neighbors_index = torch::zeros({n_atoms, max_size}, options_int);
    at::Tensor neighbors_shift = torch::zeros({n_atoms, max_size, 3}, options_int);
    at::Tensor relative_positions = torch::zeros({n_atoms, max_size, 3}, options_float);
    at::Tensor nums = torch::zeros({n_atoms}, options_int);  // Tensor to store the count of elements
    at::Tensor mask = torch::ones({n_atoms, max_size}, options_bool);  // Tensor to store the mask
    at::Tensor neighbor_species = all_species.size(0) * torch::ones({n_atoms, max_size}, options_int);

    c10::optional<at::Tensor> neighbor_scalar_attributes;
    int64_t scalar_attr_dim = 0;

    if (scalar_attributes.has_value()) {
        TORCH_CHECK(scalar_attributes->device().is_cpu(), "scalar_attributes must be on CPU");
        TORCH_CHECK(scalar_attributes->is_contiguous(), "scalar_attributes must be contiguous");
        TORCH_CHECK(scalar_attributes->size(0) == D_list.size(0), "scalar_attributes must have the same size as D_list in the first dimension");
        
        scalar_attr_dim = scalar_attributes->size(1);
        neighbor_scalar_attributes = torch::zeros({n_atoms, max_size, scalar_attr_dim}, options_float);
    }

    // Temporary array to track the current population index
    int_t* current_index = new int_t[n_atoms];
    std::fill(current_index, current_index + n_atoms, 0);  // Fill the array with zeros

    // Get raw data pointers
    int_t* i_list_ptr = i_list.data_ptr<int_t>();
    int_t* j_list_ptr = j_list.data_ptr<int_t>();
    int_t* S_list_ptr = S_list.data_ptr<int_t>();
    float_t* D_list_ptr = D_list.data_ptr<float_t>();
    int_t* species_ptr = species.data_ptr<int_t>();
    int_t* all_species_ptr = all_species.data_ptr<int_t>();

    int_t* neighbors_index_ptr = neighbors_index.data_ptr<int_t>();
    int_t* neighbors_shift_ptr = neighbors_shift.data_ptr<int_t>();
    float_t* relative_positions_ptr = relative_positions.data_ptr<float_t>();
    int_t* nums_ptr = nums.data_ptr<int_t>();
    bool* mask_ptr = mask.data_ptr<bool>();
    int_t* neighbor_species_ptr = neighbor_species.data_ptr<int_t>();

    float_t* scalar_attributes_ptr = nullptr;
    float_t* neighbor_scalar_attributes_ptr = nullptr;
    if (scalar_attributes.has_value()) {
        scalar_attributes_ptr = scalar_attributes->data_ptr<float_t>();
        neighbor_scalar_attributes_ptr = neighbor_scalar_attributes->data_ptr<float_t>();
    }
    
    int64_t all_species_size = all_species.size(0);
    
    int_t all_species_maximum = -1;
    for (int64_t k = 0; k < all_species_size; ++k) {
        if (all_species_ptr[k] > all_species_maximum) {
            all_species_maximum = all_species_ptr[k];
        }
    }
    
    int_t* mapping = new int_t[all_species_maximum + 1];
    for (int64_t k = 0; k < all_species_size; ++k) {
        mapping[all_species_ptr[k]] = k;
    }
    
    
   
    // Populate the neighbors_index, neighbors_shift, relative_positions, neighbor_species, and neighbor_scalar_attributes tensors
    for (int64_t k = 0; k < i_list.size(0); ++k) {
        int_t i = i_list_ptr[k];
        int_t j = j_list_ptr[k];
        int_t idx = current_index[i];

        if (idx < max_size) {
            neighbors_index_ptr[i * max_size + idx] = j;
            neighbor_species_ptr[i * max_size + idx] = mapping[species_ptr[j]];
            /*for (int64_t q = 0; q < all_species_size; ++q) {
                if (all_species_ptr[q] == species_ptr[j]) {
                    neighbor_species_ptr[i * max_size + idx] = q;
                    break;
                }
            }*/
            
            // Unroll the loop for better computational efficiency
            neighbors_shift_ptr[(i * max_size + idx) * 3 + 0] = S_list_ptr[k * 3 + 0];
            neighbors_shift_ptr[(i * max_size + idx) * 3 + 1] = S_list_ptr[k * 3 + 1];
            neighbors_shift_ptr[(i * max_size + idx) * 3 + 2] = S_list_ptr[k * 3 + 2];

            relative_positions_ptr[(i * max_size + idx) * 3 + 0] = D_list_ptr[k * 3 + 0];
            relative_positions_ptr[(i * max_size + idx) * 3 + 1] = D_list_ptr[k * 3 + 1];
            relative_positions_ptr[(i * max_size + idx) * 3 + 2] = D_list_ptr[k * 3 + 2];

            mask_ptr[i * max_size + idx] = false;

            if (scalar_attributes.has_value()) {
                for (int64_t d = 0; d < scalar_attr_dim; ++d) {
                    neighbor_scalar_attributes_ptr[(i * max_size + idx) * scalar_attr_dim + d] = scalar_attributes_ptr[k * scalar_attr_dim + d];
                }
            }

            current_index[i]++;
        }
    }

    // Copy current_index to nums
    for (int64_t i = 0; i < n_atoms; ++i) {
        nums_ptr[i] = current_index[i];
    }
    
    at::Tensor neighbors_pos = torch::zeros({n_atoms, max_size}, options_int);
    int_t* neighbors_pos_ptr = neighbors_pos.data_ptr<int_t>();

    // Temporary array to track the current population index
    int_t* current_index_two = new int_t[n_atoms];
    std::fill(current_index_two, current_index_two + n_atoms, 0);  // Fill the array with zeros
   
    for (int64_t k = 0; k < i_list.size(0); ++k) {
        int_t i = i_list_ptr[k];
        int_t j = j_list_ptr[k];
        for (int64_t q = 0; q < current_index[j]; ++q) {
            if (neighbors_index_ptr[j * max_size + q] == i && neighbors_shift_ptr[(j * max_size + q) * 3 + 0] == -S_list_ptr[k * 3 + 0] && neighbors_shift_ptr[(j * max_size + q) * 3 + 1] == -S_list_ptr[k * 3 + 1] && neighbors_shift_ptr[(j * max_size + q) * 3 + 2] == -S_list_ptr[k * 3 + 2]) {
                neighbors_pos_ptr[i * max_size + current_index_two[i]] = q;
                current_index_two[i]++;
            }
        }
    }

    // Clean up temporary memory
    delete[] current_index;
    delete[] current_index_two;
    
    at::Tensor species_mapped = torch::zeros({n_atoms}, options_int);
    int_t* species_mapped_ptr = species_mapped.data_ptr<int_t>();
    for (int64_t k = 0; k < n_atoms; ++k) {
        species_mapped_ptr[k] = mapping[species_ptr[k]];
    }
    
    /*for (int64_t k = 0; k < n_atoms; ++k) {
         for (int64_t q = 0; q < all_species_size; ++q) {
            if (all_species_ptr[q] == species_ptr[k]) {
                species_mapped_ptr[k] = q;
                break;
            }
        }
    }*/
    
     delete[] mapping;
    // Return the results as a vector of tensors
    if (scalar_attributes.has_value()) {
        return {neighbors_index, relative_positions, neighbor_scalar_attributes, nums, mask, neighbor_species, neighbors_pos, species_mapped};
    } else {
        return {neighbors_index, relative_positions, c10::nullopt, nums, mask, neighbor_species, neighbors_pos, species_mapped};
    }
}

// Dispatch function based on tensor types
std::vector<c10::optional<at::Tensor>> process_dispatch(at::Tensor i_list, at::Tensor j_list, at::Tensor S_list, at::Tensor D_list, 
                                                        int64_t max_size, int64_t n_atoms, at::Tensor species, c10::optional<at::Tensor> scalar_attributes,
                                                        at::Tensor all_species) {
    if (i_list.scalar_type() == at::ScalarType::Int && j_list.scalar_type() == at::ScalarType::Int &&
        S_list.scalar_type() == at::ScalarType::Int && D_list.scalar_type() == at::ScalarType::Float) {
        return process_neighbors<int32_t, float>(i_list, j_list, S_list, D_list, max_size, n_atoms, species, scalar_attributes, all_species);
    } else if (i_list.scalar_type() == at::ScalarType::Int && j_list.scalar_type() == at::ScalarType::Int &&
               S_list.scalar_type() == at::ScalarType::Int && D_list.scalar_type() == at::ScalarType::Double) {
        return process_neighbors<int32_t, double>(i_list, j_list, S_list, D_list, max_size, n_atoms, species, scalar_attributes, all_species);
    } else if (i_list.scalar_type() == at::ScalarType::Long && j_list.scalar_type() == at::ScalarType::Long &&
               S_list.scalar_type() == at::ScalarType::Long && D_list.scalar_type() == at::ScalarType::Float) {
        return process_neighbors<int64_t, float>(i_list, j_list, S_list, D_list, max_size, n_atoms, species, scalar_attributes, all_species);
    } else if (i_list.scalar_type() == at::ScalarType::Long && j_list.scalar_type() == at::ScalarType::Long &&
               S_list.scalar_type() == at::ScalarType::Long && D_list.scalar_type() == at::ScalarType::Double) {
        return process_neighbors<int64_t, double>(i_list, j_list, S_list, D_list, max_size, n_atoms, species, scalar_attributes, all_species);
    } else {
        throw std::runtime_error("Unsupported tensor types");
    }
}

// Register the function as a JIT operator
static auto registry = torch::RegisterOperators("neighbors_convert::process", &process_dispatch);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("process", &process_dispatch, "Process neighbors and return tensors, including optional scalar attributes, count tensor, mask, and neighbor_species");
}

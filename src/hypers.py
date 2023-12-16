import yaml
import warnings
import re
import inspect

def propagate_duplicated_params(provided_hypers, default_hypers, first_key, second_key, check_duplicated):
    if check_duplicated:
        if (first_key in provided_hypers.keys()) and (second_key in provided_hypers.keys()):
            raise ValueError(f"only one of {first_key} and {second_key} should be provided")

        if (first_key in default_hypers.keys()) and (second_key in default_hypers.keys()):
            raise ValueError(f"only one of {first_key} and {second_key} should be in default hypers")

    output_key, output_value = None, None
    for key in [first_key, second_key]:
        if key in provided_hypers.keys():
            output_key = key
            output_value = provided_hypers[key]
            
    if output_key is None:
        for key in [first_key, second_key]:
            if key in default_hypers.keys():
                output_key = key
                output_value = default_hypers[key]                
        
    if output_key is None:
        raise ValueError(f"{first_key} or {second_key} must be provided somewhere")
        
    return output_key, output_value

def combine_hypers(provided_hypers, default_hypers, check_duplicated):    
    duplicated_params = ['ATOMIC_BATCH_SIZE', 'STRUCTURAL_BATCH_SIZE',
                         'EPOCH_NUM', 'EPOCH_NUM_ATOMIC',
                         'SCHEDULER_STEP_SIZE', 'SCHEDULER_STEP_SIZE_ATOMIC',
                         'EPOCHS_WARMUP', 'EPOCHS_WARMUP_ATOMIC']
    
    
    for key in provided_hypers.keys():
        if key not in default_hypers.keys():
            if key not in duplicated_params:                
                raise ValueError(f"unknown hyper parameter {key}")
    
    result = {}
    
    for key in default_hypers.keys():        
        if key in provided_hypers.keys():
            if key not in duplicated_params:
                result[key] = provided_hypers[key]
        else:
            if key not in duplicated_params:
                result[key] = default_hypers[key]
   

    dupl_key, dupl_value = propagate_duplicated_params(provided_hypers, default_hypers, 'ATOMIC_BATCH_SIZE', 
                                                         'STRUCTURAL_BATCH_SIZE', check_duplicated)               
    result[dupl_key] = dupl_value
    
    
    dupl_key, dupl_value = propagate_duplicated_params(provided_hypers, default_hypers, 'EPOCH_NUM', 
                                                         'EPOCH_NUM_ATOMIC', check_duplicated)               
    result[dupl_key] = dupl_value 
    
    dupl_key, dupl_value = propagate_duplicated_params(provided_hypers, default_hypers, 'SCHEDULER_STEP_SIZE', 
                                                         'SCHEDULER_STEP_SIZE_ATOMIC', check_duplicated)               
    result[dupl_key] = dupl_value  
    
    dupl_key, dupl_value = propagate_duplicated_params(provided_hypers, default_hypers, 'EPOCHS_WARMUP', 
                                                         'EPOCHS_WARMUP_ATOMIC', check_duplicated)               
    result[dupl_key] = dupl_value   
        
        
    if (not result['USE_ENERGIES']) and (not result['USE_FORCES']):
        raise ValueError("At least one of the energies and forces should be used for fitting")
        
    if (not result['USE_ENERGIES']) or (not result['USE_FORCES']):
        if (result['ENERGY_WEIGHT'] is not None):
            warnings.warn("ENERGY_WEIGHT was provided, but in the current calculation, it doesn't affect anything since only one target of energies and forces is used")
            
    if result['USE_ADDITIONAL_SCALAR_ATTRIBUTES']:
        if result['SCALAR_ATTRIBUTES_SIZE'] is None:
            raise ValueError("scalar attributes size must be provided if use_additional_scalar_attributes == True")
            
    return result

class Hypers():
    def __init__(self):
        self.is_set = False
    
    
    def set_from_dict(self, hypers_dict):
        if self.is_set:
            raise ValueError("Hypers are already set")
        for k, v in hypers_dict.items():
            setattr(self, k, v)
        self.is_set = True
        
    @staticmethod 
    def fix_Nones_in_yaml(hypers_dict):
        for key in hypers_dict.keys():
            if (hypers_dict[key] == 'None') or (hypers_dict[key] == 'none'):
                hypers_dict[key] = None
              
    
    def load_from_file(self, path_to_hypers):
        if self.is_set:
            raise ValueError("Hypers are already set")
            
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        
        with open(path_to_hypers, 'r') as f:
            hypers = yaml.load(f, Loader = loader)
            Hypers.fix_Nones_in_yaml(hypers)
            
        self.set_from_dict(hypers)      
        
    
            
    
    def set_from_files(self, path_to_provided_hypers, path_to_default_hypers, check_duplicated = True):
        if self.is_set:
            raise ValueError("Hypers are already set")
            
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))

        with open(path_to_provided_hypers, 'r') as f:
            provided_hypers = yaml.load(f, Loader = loader)
            Hypers.fix_Nones_in_yaml(provided_hypers)
        
        with open(path_to_default_hypers, 'r') as f:
            default_hypers = yaml.load(f, Loader = loader)
            Hypers.fix_Nones_in_yaml(default_hypers)
        
        combined_hypers = combine_hypers(provided_hypers, default_hypers, check_duplicated)
        self.set_from_dict(combined_hypers)
        
       
def save_hypers(hypers, path_save):
    all_members = inspect.getmembers(hypers, lambda member:not(inspect.isroutine(member)))
    all_hypers = []
    for member in all_members:
        if member[0].startswith('__'):
            continue
        if member[0] == 'is_set':
            continue
        all_hypers.append(member)
    all_hypers = {hyper[0] : hyper[1] for hyper in all_hypers}

    with open(path_save, "w") as f:
        yaml.dump(all_hypers, f)
    
    
   

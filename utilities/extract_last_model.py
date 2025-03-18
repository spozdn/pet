import torch
import argparse

def extract_model_state_dict(input_checkpoint_path, output_model_state_dict_path):
    """
    Extracts the model state dictionary from a given checkpoint and saves it to the specified output path.

    Parameters:
    input_checkpoint_path (str): Path to the input checkpoint file.
    output_model_state_dict_path (str): Path to save the extracted model state dictionary.
    """
    # Load the checkpoint
    checkpoint = torch.load(input_checkpoint_path, map_location=torch.device('cpu'))

    # Extract the model state dictionary
    model_state_dict = checkpoint["model_state_dict"]

    # Save the model state dictionary
    torch.save(model_state_dict, output_model_state_dict_path)
    print(f"Model state dictionary has been saved to {output_model_state_dict_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Extracting the state of the model at the end of fitting and exposing it as all the other model state dicts, such as "best_val_rmse_both_model_state_dict" or "best_val_mae_both_model_state_dict".'
    )
    parser.add_argument('path_to_calc_folder', type=str, help='Path to the calc folder.')

    args = parser.parse_args()
    input_checkpoint_path = args.path_to_calc_folder + "/checkpoint"
    output_model_state_dict_path = args.path_to_calc_folder + "/last_model_state_dict"
    extract_model_state_dict(input_checkpoint_path, output_model_state_dict_path)

if __name__ == '__main__':
    main()

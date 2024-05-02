Run model
=========

After the installation, the command line script :bash:`pet_run` is available, and can be used to run fitted PET models on the new (test) structures. The usage is the following:

.. code-block:: bash

    $ pet_run <structures_path> <path_to_calc_folder> <checkpoint> <n_aug>  <batch_size> --path_save_predictions=<your_path>


where the positional arguments are:

#. structures_path: Path to an xyz file with structures
#. path_to_calc_folder : Path to a folder with a model to use
#. checkpoint: checkpoint
#. n_aug: A number of rotational augmentations to use. 
   It should be a positive integer or -1. 
   If -1, the initial coordinate system will be used, not a single random one, as in the n_aug = 1 case
#. batch_size: Batch size to use for inference. 
   It should be a positive integer or -1. 
   If -1, it will be set to the value used for fitting the provided model.

While the other options are:

* `-h, --help`:           it shows the help message and exit
* `--path_save_predictions PATH_SAVE_PREDICTIONS`:
                        Path to a folder where to save predictions.
* `--verbose`:            It shows more details
* `--dtype DTYPE`:         dtype to be used; one of 'float16', 'bfloat16', 'float32'.

Structures should be formatted in the same way as training and validation ones, as discussed in the "train model" section. The Calc folder is a path to a folder with checkpoints which is created by the "train_model.py" script. <checkpoint> refers to a specific checkpoint to be used. PET saves several checkpoints, such as the one with the best MAE in energies on validation or the best RMSE in forces on validation, which can happen on distinct epochs. Run :code:`python3 estimate_error.py --help` to see the full list. <n_aug> is a number of rotational augmentations during inference. Finally, one can optionally specify the path where predicted energies and forces are to be saved as numpy (.npy) arrays.
   
   

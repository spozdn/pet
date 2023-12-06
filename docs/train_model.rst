Train model
===========

The script src/train_model.py can be used to fit PET models. The usage is the following:

.. code-block:: bash

   $ python3 train_model.py <train_structures_path>  <val_structures_path> <provided_hypers_path> <default_hypers_path> <name_of_calculation>
       
       
Here :code:`<train_structures_path>` and :code:`<val_structures_path>` are paths to training and validation datasets. Both datasets should be contained in the xyz files readable by the `Atomic Simulation Environment package <https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read>`_. Energies should be stored in the :code:`.info` field of the corresponding `Atoms object <https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`_ by the key :code:`energy`. Forces should be stored in the :code:`.arrays` field by the key :code:`forces`. In other words, they should be accessible as:


.. code-block:: python3

    import ase.io
    structures = ase.io.read(structures_path, index = ':')
    energy_of_the_first_configuration = structures[0].info['energy']
    forces_of_the_first_configurations = structures[0].info['forces']
    
    
Please ensure that the units of energies, forces, and atomic positions are in line with each other. That is, units of forces are units of energies divided by units of atomic positions. 

It is possible to use only energies or only forces for fitting. In this case, one should specify the desired behavior setting corresponding hyperparameters, as will be discussed later. 

PET supports both crystalline and finite configurations. 

There is a lot of flexibility in constructing a Point Edge Transformer neural network. In addition, one should specify how the fitting process is performed. This results in a relatively large set of hyperparameters controlling every aspect of architecture and a fitting procedure. The complete list of hypers can be accessed at default_hypers/default_hypers.yaml. 

The good news is that to get good results, one should change only a few hypers from the default values. A user can create a YAML file with only a very few hyperparameters. Next, the training script should be supplied with a user YAML file and a YAML file with default hypers at default_hypers/default_hypers.yaml. It will result in that only hypers specified by a user being changed from the default values. For instance, to launch a calculation where only the cutoff radius differs from the default value, one should create a YAML file with one line "R_CUT: <your_value>" and launch the training script providing it with both this file and default_hypers/default_hypers.yaml.

Finally, the last argument specifies the name of the calculation. When the training script is launched, it creates a folder "results/<provided_name_of_calculation>" and stores all the checkpoints, logs, and other information there. 

A stopping criterion for the fitting procedure is either reaching a maximal number of epochs or reaching a maximal fitting time. The former can be specified either with the hyper "EPOCH_NUM", while the latter can be set with "MAX_TIME" (in seconds). After the training is done, it is possible to continue fitting from the last checkpoint if the model appears to be underfitted. One has to relaunch the training script with the same calculation name. This time all the results will be stored in a folder "results/<provided_name_of_calculation>_continuation_0". Next time it will be "continuation_1" and so on. 

`This link <https://zenodo.org/record/7967079>`_ contains a code snippet illustrating how one can read logs and plot a dependence of the validation error in energies or forces on the epoch number. It is given in the jupyter notebooks calculations/coll/plotting_energies.ipynb and calculations/coll/plotting_forces.ipynb, for example (One needs to download only the small file). Such plots show if the model has already converged or if it is desirable to continue the fitting procedure, as described above. 

Another useful file in the calculation folder is "summary.txt" which contains a few lines with the best MAE/RMSE in energies/forces on a validation dataset. 






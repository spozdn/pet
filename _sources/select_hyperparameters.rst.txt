Select hyperparameters
======================


The cutoff radius is the only hyperparameter that must be specified for a particular dataset. The selected cutoff significantly impacts the model. Its accuracy and fitting/inference times are very sensitive to this parameter.

A good starting point is to select a cutoff radius that ensures about 20 neighbors on average. This can be done by analyzing the neighbor lists for different cutoffs before launching the training script. `This <https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html>`_ is an example of a neighbor list constructor in Python.

The hyperparameter for the cutoff radius is called "R_CUT".

The second step involves adjusting the fitting duration of the model. Unlike the specification of a dataset-specific cutoff radius, this step is optional, as reasonable results can already be obtained with the default fitting duration. The time required to fit the model is a complicated function of the model's size, the size of the dataset, and the complexity of the studied interatomic interactions. The default value might be insufficient for huge datasets. As discussed in the previous section, it's possible to continue the fitting procedure if the model appears to be underfitted. One can do this by relaunching the fitting script with the same calculation name. However, the total number of epochs is only part of the equation. The other key aspect is the rate at which the learning rate decays. We use `StepLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html>`_ as a learning rate scheduler. Thus, to increase the overall fitting time, one needs to specify a larger step size which controls how fast StepLR decreases the learning rate. This can be done by specifying the hyperparameter "SCHEDULER_STEP_SIZE". 

For such hyperparameters as "SCHEDULER_STEP_SIZE", "EPOCH_NUM", "BATCH_SIZE", and "EPOCHS_WARMUP", either normal or atomic versions can be specified. Atomic versions are termed "SCHEDULER_STEP_SIZE_ATOMIC", "EPOCH_NUM_ATOMIC", "BATCH_SIZE_ATOMIC", and "EPOCHS_WARMUP_ATOMIC". Let's take batch size, for instance. It makes no sense to use the same batch size for datasets with structures of very different sizes. If one dataset contains, let's say, molecules with 10 atoms on average and the other nanoparticles with 1000 atoms on average, it makes sense to use a 100 times larger batch size in the first case. If "BATCH_SIZE_ATOMIC" is specified, the normal batch size is computed as BATCH_SIZE = BATCH_SIZE_ATOMIC / (average_number_of_atoms_in_the_training_dataset). A similar logic applies to "SCHEDULER_STEP_SIZE", "EPOCH_NUM", and "EPOCHS_WARMUP". In these cases, normal versions are obtained by division by the total number of atoms of structures in the training dataset. All the default values are given by atomic versions for better transferability.

Thus, in order to increase the step size of the learning rate scheduler by, let's say, 2 times, one can take the default value for "SCHEDULER_STEP_SIZE_ATOMIC" from the default_hypers/default_hypers.yaml and specify a value that's twice as large.

To fit the model only on energies, one can specify: "USE_FORCES: False". Specification for fitting on forces is: "USE_ENERGIES: False".







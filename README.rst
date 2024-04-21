.. inclusion-marker-preambule-start-first

.. role:: bash(code)
   :language: bash
   
PET
===

This repository contains an implementation of Point Edge Transformer (PET), interatomic machine learning potential, which achieves state-of-the-art on several datasets; see more details in [1]. PET is a graph neural network where each message-passing layer is given by an arbitrarily deep transformer. Additionally, this repository contains a proof-of-principle implementation of the Equivariant Coordinate System Ensemble (ECSE). 

++++++++++++
Installation
++++++++++++

Run :bash:`pip install .`

After the installation, the following command line scripts are available: :bash:`pet_train`, :bash:`pet_run`, and 
:bash:`pet_run_sp`. 

See the documentation for more details. 
   
.. inclusion-marker-preambule-end-first

+++++++++
Ecosystem
+++++++++

.. image:: /figures/ecosystem.svg
   :alt: Ecosystem overview
   :align: center


`LAMMPS <https://www.lammps.org/#gsc.tab=0>`_, `i-pi <https://ipi-code.org/i-pi/>`_, and `ASE.MD <https://wiki.fysik.dtu.dk/ase/ase/md.html>`_ are molecular simulation engines.

`MTM <https://github.com/lab-cosmo/metatensor-models>`_ enables the creation of a single interface for models such as PET to use them immediately in multiple simulation engines. Currently, this is implemented for LAMMPS and ASE.MD, with plans to extend it to additional engines in the future.

All MD interfaces are currently under development and are not stable. The :bash:`pet_train` and :bash:`pet_run` scripts are now semi-stable.

"sp" stands for Symmetrization Protocol and refers to ECSE.

MLIP stands for Machine Learning Interatomic Potential. Fitting is supported for both energies and/or forces (It is always recommended to use forces if they are available). The :bash:`pet_train_general_target` script is designed for fitting multidimensional targets such as eDOS. Optionally, a target may be atomic, meaning it is assigned to each atom in the structure rather than the entire atomic configuration. Derivatives are not supported by this script.

+++++++++++++
Documentation
+++++++++++++

Documentation can be found `here <https://spozdn.github.io/pet/>`_.
   
.. inclusion-marker-preambule-start-second

+++++
Tests
+++++

:bash:`cd tests && pytest .`

++++++++++
References
++++++++++

[1] Sergey Pozdnyakov, and Michele Ceriotti 2023. Smooth, exact rotational symmetrization for deep learning on point clouds. In Thirty-seventh Conference on Neural Information Processing Systems.

.. inclusion-marker-preambule-end-second

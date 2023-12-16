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

+++++++++++++
Documentation
+++++++++++++

Documentation can be found `here <https://serfg.github.io/pet/>`_.
   
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
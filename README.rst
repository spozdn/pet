.. role:: bash(code)
   :language: bash
   
PET
===

This repository contains an implementation of Point Edge Transformer (PET), interatomic machine learning potential, which achieves state-of-the-art on several datasets; see more details in [1]. PET is a graph neural network where each message-passing layer is given by an arbitrarily deep transformer. Additionally, this repository contains a proof-of-principle implementation of the Equivariant Coordinate System Ensemble (ECSE). 

++++++++++++
Installation
++++++++++++

1. Run :bash:`pip install -r requirements.txt`
2. Install `pytorch-geometric <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_ matching your version of CUDA
3. After that three command-line tools are at your disposal - src/train_model.py, src/estimate_error.py, src/estimate_error_sp.py
   
   
++++++++++
References
++++++++++

[1] Pozdnyakov, Sergey N., and Michele Ceriotti. "Smooth, exact rotational symmetrization for deep learning on point clouds." arXiv preprint arXiv:2305.19302 (2023).
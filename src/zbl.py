#from typing import Dict, List, Optional

import torch
import numpy as np
    
def get_pairwise_zbl(  zi, zj, rij) -> float:
    """Ziegler-Biersack-Littmark (ZBL) potential."""

    rc=1 #this is a hyperparameter 

    rc = torch.tensor(rc).unsqueeze(-1)
    r1 = 0.0
    p = 0.23
    # angstrom
    a0 = 0.46850
    c = torch.tensor([0.02817, 0.28022, 0.50986, 0.18175])
    d = torch.tensor([0.20162, 0.40290, 0.94229, 3.19980])

    a =  a0 / (zi **  p + zj **  p)

    da =  d.unsqueeze(-1)/a

    phi_x =  _phi(rij,c,da)

    # e * e / (4 * pi * epsilon_0) / electron_volt / angstrom
    factor=14.399645478425668 * zi * zj
    e =  _e_zbl(factor,rij,c,da)  # eV.angstrom

    # switching function
    ec =  _e_zbl(factor, rc,c,da )
    dec =  _dedr(factor, rc,c,da)
    d2ec =  _d2edr2(factor, rc,c,da)

    # coefficients are determined such that E(rc) = 0, E'(rc) = 0, and E''(rc) = 0
    A = (-3 *  dec + ( rc- r1) *  d2ec) / (( rc- r1) ** 2)
    B = (2 *  dec - ( rc- r1) *  d2ec) / (( rc- r1) ** 3)
    C = -  ec + ( rc- r1) *  dec / 2 - ( rc- r1) * ( rc- r1) *  d2ec / 12 

    e += A / 3 * ((rij- r1) ** 3) + B / 4 * ((rij- r1) ** 4) + C
    return e

def _phi(r,c,da):
    phi = torch.sum( c.unsqueeze(-1) * torch.exp(-r *  da ),dim=0 )
    return phi

def _dphi(r,c,da):
    dphi = torch.sum(- c.unsqueeze(-1) *  da  * torch.exp(-r *  da ),dim=0)
    return dphi

def _d2phi(r,c,da):
    d2phi = torch.sum( c.unsqueeze(-1)  * ( da  ** 2) * torch.exp(- rc *  da ),dim=0)
    return d2phi
    
def _e_zbl(factor, r,c,da):
    phi =  _phi(r,c,da)
    ret = factor / r *phi 
    return ret

def _dedr(factor,r,c,da):
    phi =  _phi(r,c,da)
    dphi =  _dphi(r,c,da)
    ret = factor / r * (-phi / r + dphi)
    return ret

def _d2edr2(factor,r,c,da):
    phi =  _phi(r,c,da)
    dphi =  _dphi(r,c,da)
    d2phi =  _d2phi(r,c,da)

    ret = factor / r * (d2phi - 2 / r * dphi + 2 * phi / (r ** 2))
    return ret



#rc=1
#r=torch.linspace(0.1, rc, steps=100, requires_grad=True)
#print(r)
#zi=torch.ones(len(r))*10
#zj=torch.ones(len(r))*10
#
#energies=get_pairwise_zbl(zi, zj, r)
#
#import matplotlib.pyplot as plt
#fig,ax = plt.subplots(figsize=(6.75,6.75))
#plt.plot(r.detach(), [e.detach() for e in energies],'o')
#energy=torch.sum(energies)
#energy.backward()
#plt.plot(r.detach(), r.grad.detach(), color='r',marker='d')
#
#
##ax.set_yscale('log')
#
##ax.set_ylim(1e-4, 1e4)
#
#plt.show()






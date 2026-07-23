# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:55:11 2024

@author: heatherquayle

Adapted from codes by Jack M. Hemingway (University of Edinburgh, In-person communication) 
and J. Rein et al. (ref: doi.org/10.1002/anie.202218213)

"""
### script to extract molecular descriptors from sheet containing SMILES ###

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import rdkit
from rdkit import Chem
import math
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import GraphDescriptors as gdesc
from rdkit.Chem import Lipinski
from rdkit.Chem import EState
from rdkit.Chem import MolSurf
from collections import OrderedDict

def counter(molecule, fg):
    molecule_mol = Chem.MolFromSmiles(molecule)
    funcgrp = Chem.MolFromSmarts(fg)
    output = (len(molecule_mol.GetSubstructMatches(funcgrp)))
    return output

def chemdescriptor(molecule):
    # calculate molecular formula weight from SMILES string
    MolWt = Descriptors.MolWt(Chem.MolFromSmiles(molecule))
    HeavyMolWt = Descriptors.HeavyAtomMolWt(Chem.MolFromSmiles(molecule))
    numH = math.floor(MolWt-HeavyMolWt)
    # deduce atom count for the molecule
    numC = counter(molecule, '[#6]')
    numN = counter(molecule, '[#7]')
    numO = counter(molecule, '[#8]')
    # calculate oxygen balance for the molecule
    OxBal = (-1600/MolWt)*(2*numC+(numH/2)-numO)
    # count smarts needed for h-bond ratio calculation
    no2 = counter(molecule, '[$([NX3](=O)=O),$([NX3+](=O)[O-])]')
    nh2 = counter(molecule, '[NH2]')
    co = counter(molecule, '[#6X3]=[OX1]')
    coc = counter(molecule, '[$([#8])D2]([#6])[#6]')
    con = counter(molecule, '[$([#8])D2]([#7])[#6]')
    nhr2 = counter(molecule, '[#7X3H1]')
    ringn = counter(molecule, '[#7D2H0]')
    oh = counter(molecule, '[$(O)H]')
    nr3 = counter(molecule, '[$([#7]);X3;!$(N[O]);H0]')
    
    # calculate h-bond ratio
    donors = (2*nh2) + nhr2 + oh
    acceptors = (2*no2) + nh2 + co + coc + con + nhr2 + ringn + oh + nr3
    
    if acceptors == 0:
        hbondratio = 0
        # TAPT hbond acceptrors fail 
        print('Warning: no h-bond acceptors detected for molecule with SMILES ' + molecule + '. H-bond ratio set to 0.')
    else:
        hbondratio = (donors**2)/acceptors

    # define other features not from smarts queries
    rotbonds = Lipinski.NumRotatableBonds(Chem.MolFromSmiles(molecule))
    heteroatoms = Lipinski.NumHeteroatoms(Chem.MolFromSmiles(molecule))
    totrings = Lipinski.RingCount(Chem.MolFromSmiles(molecule))
    aromrings = Lipinski.NumAromaticRings(Chem.MolFromSmiles(molecule))
    alirings = Lipinski.NumAliphaticRings(Chem.MolFromSmiles(molecule))
    VSA8 = EState.EState_VSA.VSA_EState8(Chem.MolFromSmiles(molecule))
    VSA5 = MolSurf.SMR_VSA5(Chem.MolFromSmiles(molecule))
    TPSA = MolSurf.TPSA(Chem.MolFromSmiles(molecule))

    # extract values needed for KMF calculation & calculate
    K1 = gdesc.Kappa1(Chem.MolFromSmiles(molecule))
    K2 = gdesc.Kappa2(Chem.MolFromSmiles(molecule))
    heavy_atom_count=Descriptors.HeavyAtomCount(Chem.MolFromSmiles(molecule))
    kmf=K1*K2/heavy_atom_count

    # count smarts for trigger bonds
    class1 = no2
    cnali = counter(molecule, '[$([#7]-[#6]);!$(N[O])]') 
    coali = counter(molecule, '[C;!R]-[OH0]')
    calinaro = counter(molecule, '[C;!R]-[#7;R]')
    nnh2 = counter(molecule, '[#7]~[NH2]')
    cp = counter(molecule, '[#6]-[#15]')
    class2 = coali + calinaro + nnh2 + cp + cnali

    # function can return values listed below
    return {
        'rotbonds': rotbonds,
        'heteroatoms': heteroatoms,
        'totrings': totrings,
        'aromrings': aromrings,
        'alirings': alirings,
        'VSA8': VSA8,
        'VSA5': VSA5,
        'TPSA': TPSA,
        'MolWt': MolWt,
        'numC': numC,
        'numN': numN,
        'numO': numO,
        'OxBal': OxBal,
        'hbondratio': hbondratio,
        'kmf': kmf,
        'class1': class1,
        'class2': class2,
    }


# # # MAIN JOB # # #
if __name__ == "__main__":
    # example use 
    data = pd.read_csv("../01_DATASETS/CD/CD.csv")
    smiles = data["SMILES"].tolist()
    descriptors = [chemdescriptor(s) for s in smiles]
    df = pd.DataFrame(descriptors)
    print(df.head())
    
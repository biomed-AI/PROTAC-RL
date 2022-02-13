#!/usr/bin/env python
from __future__ import print_function, division
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.QED import qed
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit import DataStructs
import joblib
from sklearn import ensemble

import time
import pickle
import re
import sys
import threading
import pexpect
import importlib
import multiprocessing
from onmt.reinforce.score_util import calc_SC_RDKit_score

from rdkit import rdBase, RDLogger
RDLogger.DisableLog('rdApp.*')  # https://github.com/rdkit/rdkit/issues/2683
rdBase.DisableLog('rdApp.error')

"""Scoring function should be a class where some tasks that are shared for every call
   can be reallocated to the __init__, and has a __call__ method which takes a single SMILES of
   argument and returns a float. A multiprocessing class will then spawn workers and divide the
   list of SMILES given between them.

   Passing *args and **kwargs through a subprocess call is slightly tricky because we need to know
   their types - everything will be a string once we have passed it. Therefor, we instead use class
   attributes which we can modify in place before any subprocess is created. Any **kwarg left over in
   the call to get_scoring_function will be checked against a list of (allowed) kwargs for the class
   and if a match is found the value of the item will be the new value for the class.

   If num_processes == 0, the scoring function will be run in the main process. Depending on how
   demanding the scoring function is and how well the OS handles the multiprocessing, this might
   be faster than multiprocessing in some cases."""


class CLOGP():
    """Scores structures based on ClogP."""

    kwargs = ["src", "ref", "goal"]
    src = ""
    ref = ""
    goal_ClogP = 3

    def __init__(self):
        self.src_new = remove_dummys(self.src)
        self.goal_ClogP = self.goal


    def __call__(self, smile):
        # gtruth_structure is the godden structure, smile is generated by agent
        mol = Chem.MolFromSmiles(smile)
        if mol:
            isstandard = juice_is_standard_contains_fregments(smile, self.src_new)
            if isstandard:
                mol_ClogP = Chem.Crippen.MolLogP(mol)
                # Rclogp = max(0.0, 1 - (1/10) * ((mol_ClogP-self.goal_ClogP)**2))
                Rclogp = max(0.0, 1 - (1/6)*abs(mol_ClogP - self.goal_ClogP))
                return Rclogp
            else:
                return 0.0
        return 0.0


class MW():
    """Scores structures based on MW."""

    kwargs = ["src", "ref", "goal"]
    src = ""
    ref = ""
    goal_molecular_weight = 1100

    def __init__(self):
        self.src_new = remove_dummys(self.src)
        self.goal_molecular_weight = self.goal

    def __call__(self, smile):
        # gtruth_structure is the godden structure, smile is generated by agent
        mol = Chem.MolFromSmiles(smile)
        if mol:
            isstandard = juice_is_standard_contains_fregments(smile, self.src_new)

            if isstandard:
                weight = Descriptors.MolWt(mol)
                Rmw = max(0, 1 - 10e-6 * ((weight - self.goal_molecular_weight) ** 2))
                return Rmw
            else:
                return 0.0
        return 0.0


class linker_length():
    """Scores structures based on linker_length"""

    kwargs = ["src", "ref", "goal"]
    src = ""
    ref = ""
    goal = 0

    def __init__(self):
        self.src_new = remove_dummys(self.src)
        self.goal_length = self.goal
        # pass

    def __call__(self, smile):
        # gtruth_structure is the godden structure, smile is generated by agent
        mol = Chem.MolFromSmiles(smile)
        if mol:
            isstandard = juice_is_standard_contains_fregments(smile, self.src_new)
            if isstandard:
                linker = get_linker(smile, self.src_new)
                if linker:
                    linker_mol = Chem.MolFromSmiles(linker)
                    linker_site_idxs = [atom.GetIdx() for atom in linker_mol.GetAtoms() if atom.GetAtomicNum() == 0]
                    linker_length = len(Chem.rdmolops.GetShortestPath(linker_mol, \
                                                                    linker_site_idxs[0], linker_site_idxs[1])) - 2
                    return max(0.0, 1 - 1/5 * abs(linker_length-self.goal_length))
                else:
                    return 0.0
            else:
                return 0.0

        return 0.0


class PK():
    kwargs = ["src", "ref", "goal"]
    src = ""
    ref = ""
    goal = 0

    def __init__(self):
        self.src_new = remove_dummys(self.src)
        self.PK = self.goal
        # pass

    def __call__(self, smile):
        # gtruth_structure is the godden structure, smile is generated by agent
        mol = Chem.MolFromSmiles(smile)
        if mol:
            isstandard = juice_is_standard_contains_fregments(smile, self.src_new)
            #isstandard = True
            if isstandard:
                myMolLogP = Crippen.MolLogP(mol)
                NAR = rdMolDescriptors.CalcNumAromaticRings(mol)
                NROTB = rdMolDescriptors.CalcNumRotatableBonds(mol)
                score = max(0, 1-(1/8)*abs((abs(myMolLogP-3) + NAR + NROTB)-self.PK))
                return score
            else:
                return 0.0

        return 0.0


def juice_is_standard_contains_fregments(gen, frags):
    "input generated molecules and the starting fragments of original molecules \
     return to the generated linker and  the two linker sites in fragments"

    m = Chem.MolFromSmiles(gen)
    matches = m.GetSubstructMatches(Chem.MolFromSmiles(frags))
    if matches:

        atoms = m.GetNumAtoms()
        for index,match in enumerate(matches):
            atoms_list = list(range(atoms))
            for i in match:
                atoms_list.remove(i)

            linker_list = atoms_list.copy()
            for i in atoms_list:
                atom = m.GetAtomWithIdx(i)
                for j in atom.GetNeighbors():
                    linker_list.append(j.GetIdx())
            linker_list = list(set(linker_list))
            sites = list(set(linker_list).difference(set(atoms_list)))

            if len(sites) == 2:
                return True
    else:
        return False

    return False

def remove_dummys(smi_string):
    try:
        smi = Chem.MolToSmiles(Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(smi_string), \
                                                                       Chem.MolFromSmiles('*'), \
                                                                       Chem.MolFromSmiles('[H]'), True)[0]))
    except:
        smi = ""

    return smi


def get_linker(gen, frags):
    m = Chem.MolFromSmiles(gen)
    matchs = m.GetSubstructMatches(Chem.MolFromSmiles(frags))

    for match in matchs:
        # remove fragments
        atoms = m.GetNumAtoms()
        atoms_list = list(range(atoms))
        for i in match:
            atoms_list.remove(i)

        linker_list = atoms_list.copy()

        # add sites
        for i in atoms_list:
            atom = m.GetAtomWithIdx(i)
            for j in atom.GetNeighbors():
                linker_list.append(j.GetIdx())

        linker_list = list(set(linker_list))
        sites = list(set(linker_list).difference(set(atoms_list)))

        # get linking bonds
        bonds = []
        for i in sites:
            atom = m.GetAtomWithIdx(i)
            for j in atom.GetNeighbors():
                if j.GetIdx() in atoms_list:
                    b = m.GetBondBetweenAtoms(i, j.GetIdx())
                    bonds.append(b.GetIdx())
        bonds = list(set(bonds))

        if not bonds:
            return ""

        # get the linker which has two "*"
        bricks = Chem.FragmentOnBonds(m, bonds)  # dummyLabels=labels
        smi = Chem.MolToSmiles(bricks)
        pattern = re.compile(r"\[\d+\*?\]")
        for s in smi.split("."):
            count = pattern.findall(s)
            if len(count) == 2:
                s = s.replace(count[0], "[*]")
                linker_smi = s.replace(count[1], "[*]")
                try:
                    linker_smi = Chem.MolToSmiles(Chem.MolFromSmiles(linker_smi))
                except:
                    print(gen, frags, linker_smi)
                return linker_smi


class Worker():
    """A worker class for the Multiprocessing functionality. Spawns a subprocess
       that is listening for input SMILES and inserts the score into the given
       index in the given list."""
    def __init__(self, scoring_function=None, **kwargs):
        """The score_re is a regular expression that extracts the score from the
           stdout of the subprocess. This means only scoring functions with range
           0.0-1.0 will work, for other ranges this re has to be modified."""

        func_class = getattr(sys.modules[__name__], scoring_function)

        for key, value in kwargs.items():
            if key in func_class.kwargs:
                setattr(func_class, key, value)
        self.proc = func_class()


    def __call__(self, smile):
        return self.proc(smile)

class Multiprocessing():
    """Class for handling multiprocessing of scoring functions. OEtoolkits cant be used with
       native multiprocessing (cant be pickled), so instead we spawn threads that create
       subprocesses."""
    def __init__(self, num_processes=None, scoring_function=None, **kwargs):
        self.n = num_processes
        self.worker = Worker(scoring_function=scoring_function, **kwargs)


    def __call__(self, smiles):
        pool = multiprocessing.Pool(processes=self.n)
        scores = pool.map(self.worker, smiles)
        pool.close()
        pool.join()
        return np.array(scores, dtype=np.float32)


class Singleprocessing():
    """Adds an option to not spawn new processes for the scoring functions, but rather
       run them in the main process."""
    def __init__(self, scoring_function=None):
        self.scoring_function = scoring_function()

    def __call__(self, smiles):
        scores = [self.scoring_function(smile) for smile in smiles]
        return np.array(scores, dtype=np.float32)

def get_scoring_function(scoring_function, num_processes=None, **kwargs):
    """Function that initializes and returns a scoring function by name"""
    scoring_function_classes = [PK, linker_length, CLOGP, MW]
    scoring_functions = [f.__name__ for f in scoring_function_classes]
    scoring_function_class = [f for f in scoring_function_classes if f.__name__ == scoring_function][0]

    if scoring_function not in scoring_functions:
        raise ValueError("Scoring function must be one of {}".format([f for f in scoring_functions]))

    for k, v in kwargs.items():
        if k in scoring_function_class.kwargs:
            setattr(scoring_function_class, k, v)

    if num_processes == 0:
        return Singleprocessing(scoring_function=scoring_function_class)

    return Multiprocessing(scoring_function=scoring_function, num_processes=num_processes, **kwargs)



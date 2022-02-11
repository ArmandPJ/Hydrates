# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 19:51:51 2021

@author: Armand Parada Jackson
"""

# Practicing with ASE
import ase
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.build import fcc111, add_adsorbate
from tkinter import *
from tkinter import ttk
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from scipy import special
from math import sqrt

atom_list = ase.io.read('hydrate.pdb',format='proteindatabank')
positions = atom_list.get_positions()

print(positions)
print(type(positions))
print(positions[0].size)
print(positions[0][1])

array = np.zeros([2])
print(array)
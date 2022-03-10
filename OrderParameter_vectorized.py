# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 18:03:01 2021

@author: Armand Parada Jackson
"""

# Program to perform order parameter calculations on input coordinates

#from mpmath import * 
import sys
import warnings
import os
import time
import numpy as np
#import matplotlib.pyplot as plt
from scipy import special
from math import sqrt
import ase # just import what you need
from ase import io
from ase import Atoms
from pathlib import Path

warnings.simplefilter("ignore")
clear = lambda: os.system('clear')

#sys.path.append(".") # add this directory as a path for modules
#from neighbormanager import Neighborhood

#input_path = str(sys.argv[1])
#L = int(sys.argv[2])

directory = "/home/armandpj/Work/Hydrates/examples"

#file_name = input("Enter file name: ")
file_name = "hydrate.pdb"
input_path = Path(directory)/file_name

if not input_path.exists():
    print(f"The path {input_path} does not exist!")
    sys.exit(1)
    
#L = int(input("Enter L value (must be even): "))
L = 6

start_time = time.process_time()

r0 = 3.0 # baseline NN distance, in angstroms
rnn = 1.2*r0 # nearest neighbor threshold distance, in factor of r0
rnn2 = rnn**2 # square of neighbor threshold distance

file_object = open(input_path, "r")
file_string = file_object.read()
frames_total = file_string.count('frame')
num_frames = frames_total
frame_list = np.arange(1, num_frames+1, 1)

file_object.seek(0,0)

QL_list = []

for iframe in range(num_frames):
    
    print(f"Current Frame is: {iframe+1}/{num_frames}\n")
    
    # stores Atoms array in fileData
    file_data = ase.io.read(input_path,index=iframe,format='proteindatabank')
    symbol_data = np.array(file_data.get_chemical_symbols())
    all_atom_positions: np.ndarray = file_data.get_positions()
    
    # Filtering
    all_atom_positions = all_atom_positions[(symbol_data != 'H')] # filter out hydrogen atoms from all_atom_positions
    all_atom_positions = all_atom_positions[all_atom_positions[:,2].argsort()] # sort the array by the z coordinate

    z_min = 39 # narrow down the array to a layer of atoms between z = 39 and z = 58
    z_max = 60
    
    # only store atoms with z between 39 and 58 in layer_atom_positions
    layer_atom_positions = all_atom_positions[(all_atom_positions[:,2] >= z_min) & (all_atom_positions[:,2] <= z_max)]
    
    
    # Get useful lengths and dimensions of arrays for printing, later use in loops, etc.
    layer_num_atoms, num_coords = layer_atom_positions.shape
    total_atoms = len(all_atom_positions)
    print(f"Length of all_atom_positions:    {total_atoms}")
    print(f"Length of layer_atom_positions:  {layer_num_atoms}\n")

    # Get the width of the hydrate in x, y, z
    maxima = np.max(all_atom_positions, 0)
    minima = np.min(all_atom_positions, 0)
    # maxima and minima are arrays with elements of the maximum or minimum values of each column in all_atom_positions
    coordinate_widths = maxima - minima
    
    # Takes input of array arr and # of columns col, returns a copy of the array that has been normalized per column (each column has its own normalization constant)
    def normalize_by_column(arr, col):
            _array = arr
            _array[:, col] = (_array[:,col] - np.min(_array[:,col])) / (np.max(_array[:,col]) - np.min(_array[:,col]))
            return _array



    # Find and store Nearest Neighbors
    potential_neighbor_vectors = np.empty((0,3))

    for i in range(layer_num_atoms):
        # Narrow the search to atoms whose z component alone is within rnn distance
        atoms_to_check = all_atom_positions[(abs(all_atom_positions[:,2] - layer_atom_positions[i,2]) <= rnn)]

        diff_vectors = atoms_to_check - layer_atom_positions[i]
        potential_neighbor_vectors = np.append(potential_neighbor_vectors, diff_vectors, axis=0)

        clear()
        print(f"{i+1}/{layer_num_atoms} atoms")

    nearest_neighbor_vectors = potential_neighbor_vectors[(np.linalg.norm(potential_neighbor_vectors[:], axis=1) <= rnn)
                                                            & (np.linalg.norm(potential_neighbor_vectors[:], axis=1) != 0)]
    num_neighbor_vectors = len(nearest_neighbor_vectors)
    print(f"Nearest neighbor vectors length: {num_neighbor_vectors}")
    print(nearest_neighbor_vectors)

    #TODO: Add periodicity conditions using np.around (like NINT) in atoms_to_check?

    # Calculate QL, a bond order parameter which is averaged over all nearest neighbor vectors
    QLM = []
    QL = 0.0

    # calculates QLM, part of the QL equation, given vector r_vector and quantum numbers _m, _L
    def calc_QLM(r_vector, _m, _L):
        #print(f"QLM list: {self.QLM}")
        r_mag = np.linalg.norm(r)
        if(r[0] == 0.0):
            if(r[1] > 0): theta = np.pi/2
            elif(r[1] < 0): theta = 3*(np.pi/2)
        else: theta = np.arctan(r[1]/r[0]) # azimuthal angle
        if(theta < 0): theta += 2*np.pi # theta must be between 0 and 2*pi

        phi = np.arccos(r[2]/r_mag) # polar/colatitudinal angle
        if(phi < 0): phi += np.pi # phi must be between 0 and pi
        return special.sph_harm(_m, _L, theta, phi)
    
    # Performs calculations to obtain QL
    for m in range (-L, L+1):
        for r in nearest_neighbor_vectors:
            QLM.append(calc_QLM(r, m, L))
        QL += abs((np.sum(QLM)/num_neighbor_vectors))**2
    
    QL = np.sqrt(QL*((4*np.pi)/(2*L + 1)))
    QL_list.append(QL)
    print(f"Q{L} value for Frame {iframe+1}: {QL_list[iframe]}\n")

end_time = time.process_time()
elapsed_time = end_time - start_time

file_object.close()

print()
print(f"Elapsed time was {elapsed_time} seconds.")

# Writing the data to a text file:
with open(f'./output_files/Q_output_{file_name}.txt', 'w') as f:
    f.write(f'Bond Order Parameters for {file_name}\n')
    f.write(f'Frame          Q{L}\n')
    for i in range(len(QL_list)):
        f.write(f'{frame_list[i]}{QL_list[i]:10.4f}\n')


# Pseudocode for math:
# select an atom
# define a vector r from center of that atom to another
# check if nearest_neighbor (within rnn); if so, move onto math / if not, skip to next atom
# math: use vector r from atom to neighbor to calc. angles between nearest neighbors
# plug in angles to spherical harmonic equation to get Qlm
# average Qlm over all bonds (nearest-neighbor pairs) in sample
# plug average into Eq. 1.3 from Steinhardt to calculate Ql
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
import matplotlib.pyplot as plt
from scipy import special
from math import sqrt
import ase # just import what you need
from ase import io
from ase import Atoms
from pathlib import Path

warnings.simplefilter("ignore")

#sys.path.append(".") # add this directory as a path for modules
#from neighbormanager import Neighborhood

#input_path = str(sys.argv[1])
#L = int(sys.argv[2])

directory = "/home/armandpj/Work/Programs/examples"

#file_name = input("Enter file name: ")
file_name = "hydrate.pdb"
input_path = Path(directory)/file_name

if not input_path.exists():
    print(f"The path {input_path} does not exist!")
    sys.exit(1)
    
#L = int(input("Enter L value (must be even): "))
L = 6

start_time = time.process_time()

r0 = 2.75 # baseline NN distance
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
    
    print(f"Current Frame is: {iframe+1}/{num_frames}")
    
    # stores Atoms array in fileData
    file_data = ase.io.read(input_path,index=iframe,format='proteindatabank')
    symbol_data = np.array(file_data.get_chemical_symbols())
    all_atom_positions: np.ndarray = file_data.get_positions()
    
    # Filtering
    all_atom_positions = all_atom_positions[(symbol_data != 'H')] # filter out hydrogen atoms from all_atom_positions
    all_atom_positions = all_atom_positions[all_atom_positions[:,2].argsort()] # sort the array by the z coordinate

    z_min = 39 # narrow down the array to a layer of atoms between z = 39 and z = 58
    z_max = 58
    
    # only store atoms with z between 39 and 58 in layer_atom_positions
    layer_atom_positions = all_atom_positions[(all_atom_positions[:,2] >= z_min) & (all_atom_positions[:,2] <= z_max)]
    
    
    # Get useful lengths and dimensions of arrays for printing, later use in loops, etc.
    layer_num_atoms, num_coords = layer_atom_positions.shape
    total_atoms = len(all_atom_positions)
    print(f"Length of all_atom_positions:    {total_atoms}")
    print(f"Length of layer_atom_positions: {layer_num_atoms}")


    # Find and store Nearest Neighbors
    nearest_neighbor_vectors = np.empty((0, 3))

    dr = 0.0 # distance between two atoms being checked
    vec = np.empty(0) # vector from atom i to atom j
    vec_norm = 0.0
    for i in range(layer_num_atoms):
        for j in range(i, total_atoms):
            if(j < total_atoms):
                # Only perform calculations if the difference in z-coordinate is within rnn distance
                if(abs(layer_atom_positions[i][2] - all_atom_positions[j][2]) <= rnn):
                    vec = all_atom_positions[j] - layer_atom_positions[i]
                    vec_norm = np.linalg.norm(vec)
                    if(vec_norm <= rnn and vec_norm != 0):
                        # If vec is a nearest neighor vector, then so is -vec. This allows the loop to
                        # only check atoms sequentially rather than backtracking
                        nearest_neighbor_vectors = np.append(nearest_neighbor_vectors, [vec], axis=0)
                        nearest_neighbor_vectors = np.append(nearest_neighbor_vectors, [-1*vec], axis=0)

    num_neighbor_vectors = len(nearest_neighbor_vectors)
    print(f"Nearest neighbor vectors length: {num_neighbor_vectors}")
    print(nearest_neighbor_vectors)
    #sys.exit(1)

    
    # Calculate QL, a bond order parameter which is averaged over all nearest neighbor pairs in a neighborhood (then average over all neighborhoods)
    QLM = np.empty(0)
    QL = 0.0

    # calculates QLM given quantum numbers m & L for all nearest neighbor vectors in this neighborhood, then averages QLM values and returns the square of the absolute value
    def calc_QLM(r_vector, _m, _L):
        #print(f"QLM list: {self.QLM}")
        r_mag = np.linalg.norm(r)
        if(r[0] < 0.001):
            r[0] = 0.001
        theta = np.arctan(r[1]/r[0]) # azimuthal angle
        if(theta < 0): theta += 2*np.pi
        phi = np.arccos(r[2]/r_mag) # polar/colatitudinal angle
        if(phi < 0): phi += np.pi
        return special.sph_harm(_m, _L, theta, phi)
    
    for m in range (-L, L+1):
        for r in nearest_neighbor_vectors:
            QLM = np.append(QLM, calc_QLM(r, m, L))
        QL += abs((np.sum(QLM)/num_neighbor_vectors))**2
    
    QL = np.sqrt(QL*((4*np.pi)/(2*L + 1)))
    QL_list.append(QL)
    print(f"Q{L} value for Frame {iframe+1}: {QL_list[iframe]}\n")

end_time = time.process_time()
elapsed_time = end_time - start_time

file_object.close()

#print("List of lines/atoms read:\n")
#print(atom_list)
#print()
#print("List of coordinates for each atom:\n")
#print(layer_atom_positions)
#print()
#print("List of nearest neighbors (by list index) for each atom:\n")
#print(nn_list)
print()
#print(f"Calculated value of Q{L}: {QL}\n")
print()
print(f"Elapsed time was {elapsed_time} seconds.")

# Writing the data to a text file:
with open(f'./output_files/Q_output_{file_name}.txt', 'w') as f:
    f.write(f'Bond Order Parameters for {file_name}\n')
    f.write(f'Frame          Q{L}\n')
    for i in range(len(QL_list)):
        f.write(f'{frame_list[i]}              {QL_list[i]:.4f}\n')


# Pseudocode for math:
# select an atom
# define a vector r from center of that atom to another
# check if nearest_neighbor (within rnn); if so, move onto math / if not, skip to next atom
# math: use vector r from atom to neighbor to calc. angles relative to coordinate frame
# plug in angles to spherical harmonic equation to get Qlm
# average Qlm over all bonds (nearest-neighbor pairs) in sample
# plug average into Eq. 1.3 from Steinhardt to calculate Ql
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

r0 = 5.00 # baseline NN distance (formerly the diameter of a water molecule, 2.75 Angstroms)
rnn = 1.2*r0 # nearest neighbor threshold distance, in factor of r0
rnn2 = rnn**2 # square of neighbor threshold distance

file_object = open(input_path, "r")
file_string = file_object.read()
framesTotal = file_string.count('frame')
num_frames = framesTotal
frame_list = np.linspace(1, num_frames+1, num_frames)

file_object.seek(0,0)

QL_list = []

for iframe in range(num_frames):
    
    print(f"Current frame is: {iframe+1}")
    
    # stores Atoms array in fileData
    fileData = ase.io.read(input_path,index=iframe,format='proteindatabank')
    symbolData = np.array(fileData.get_chemical_symbols())
    all_atom_positions: np.ndarray = fileData.get_positions()
    
    # Filtering
    all_atom_positions = all_atom_positions[(symbolData != 'H')] # filter out hydrogen atoms from all_atom_positions
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

    atoms_per_nn_distance = int(np.floor(layer_num_atoms/rnn))
    print(f"Atoms per nn_distance: {atoms_per_nn_distance}")

    # Find and store Nearest Neighbors
    nearest_neighbor_vectors = np.empty((0, 3))

    dr = 0.0 # distance between two atoms being checked
    vec = np.empty((0, 3)) # vector from atom i to atom j
    index_radius_lower = 0 # determines the range of indices that j will scan
    index_radius_upper = 0
    for i in range(layer_num_atoms):
        index_radius_lower = i - atoms_per_nn_distance
        index_radius_upper = i + atoms_per_nn_distance + 1
        for j in range(index_radius_lower, index_radius_upper):
            if(0 <= j < total_atoms):
                #vec = all_atom_positions[j] - layer_atom_positions[i]
                dx = all_atom_positions[j][0] - layer_atom_positions[i][0]
                dy = all_atom_positions[j][1] - layer_atom_positions[i][1]
                dz = all_atom_positions[j][2] - layer_atom_positions[i][2]
                vec = np.array([dx, dy, dz])
                print(f"\nj atom vector: {all_atom_positions[j]}")
                print(f"i atom vector: {layer_atom_positions[i]}")
                print(f"vec = {vec}")
                dr = sqrt(np.sum(np.square(vec)))
                print(f"dr = {dr}")
                print(f"nn_distance = {rnn}")
                print(f"Difference = {abs(dr-rnn)}\n")
                print(f"Atoms per nn_distance: {atoms_per_nn_distance}")
                #dr = np.linalg.norm(vec, axis=0)
                # ensure we are NOT checking if an atom is its own nearest neighbor (dr can't be 0)
                if(dr <= rnn): # append the vector vec to nearest_neighbor_vectors iff vec's magnitude is within the threshold rnn
                    nearest_neighbor_vectors = np.append(nearest_neighbor_vectors, [vec], axis=0)
                    #print(f"Vec: {vec}")

    #print(f"Nearest neighbor vectors length: {len(nearest_neighbor_vectors)}")
    #print(nearest_neighbor_vectors)
    sys.exit(1)

    
    # Calculate QL, a bond order parameter which is averaged over all nearest neighbor pairs in a neighborhood (then average over all neighborhoods)
    QLM = []
    QL = 0.0
    
    for m in range (-L, L+1):
        for i in range(num_neighborhoods):
            QL += neighborhood_list[i].calc_QLM(m, L)
    
    QL = np.sqrt(QL*((4*np.pi)/(2*L + 1)))
    QL_list.append(QL)
    print(f"Q{L} value for this frame: {QL_list[iframe]}\n")

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

fig, ax = plt.subplots()
L_string = "Q" + str(L)
_title = "Value of " + L_string + " Frame-by-Frame"
ax.plot(frame_list, QL_list, linestyle='dotted', markersize=8)
ax.scatter(frame_list, QL_list, color='r')
ax.set(title=_title, xlabel='Frame', ylabel=L_string)
plt.show()


# Pseudocode for math:
# select an atom
# define a vector r from center of that atom to another
# check if nearest_neighbor (within rnn); if so, move onto math / if not, skip to next atom
# math: use vector r from atom to neighbor to calc. angles relative to coordinate frame
# plug in angles to spherical harmonic equation to get Qlm
# average Qlm over all bonds (nearest-neighbor pairs) in sample
# plug average into Eq. 1.3 from Steinhardt to calculate Ql
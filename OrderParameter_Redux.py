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

sys.path.append(".") # add this directory as a path for modules
from neighbormanager import Neighborhood

#input_path = str(sys.argv[1])
#L = int(sys.argv[2])

directory = "C:/Users/Armand/Desktop/College/UNT/Research/MateriaLab/Programs"

file_name = input("Enter file name: ")
#file_name = "hydrate.pdb"
input_path = Path(directory)/file_name

if not input_path.exists():
    print(f"The path {input_path} does not exist!")
    sys.exit(1)
    
L = int(input("Enter L value (must be even): "))
#L = 6

start_time = time.process_time()

r0 = 2.75 # diameter of water molecule in angstroms; in general,
          # first peak in radial distribution (Steinhardt)
rnn = 1.2*r0 # nearest neighbor threshold distance, in factor of r0
rnn2 = rnn**2 # square of neighbor threshold distance

file_object = open(input_path, "r")
file_string = file_object.read()
framesTotal = file_string.count('frame')
num_frames = framesTotal
frame_list = np.arange(1, num_frames+1, 1)

file_object.seek(0,0)

QL_list = []

# Loop through a .pdb file frame by frame
for iframe in range(num_frames):
    
    print(f"Current frame is: {iframe+1}/{num_frames}")
    
    # stores Atoms array in fileData
    fileData = ase.io.read(input_path,index=iframe,format='proteindatabank')
    symbolData = np.array(fileData.get_chemical_symbols())
    all_atom_positions: np.ndarray = fileData.get_positions()
    
    # Filtering
    all_atom_positions = all_atom_positions[(symbolData != 'H')] # filter out hydrogen atoms from all_atom_positions
    all_atom_positions = all_atom_positions[all_atom_positions[:,2].argsort()] # sort the array by the z coordinate
    
    # only store atoms with z between 39 and 58 in important_atom_positions *** (NEED TO: take parameters to set these boundaries manually) ***
    important_atom_positions = all_atom_positions[(all_atom_positions[:,2] >= 39) & (all_atom_positions[:,2] <= 60)]
    
    
    # Get useful lengths and dimensions of arrays for printing, later use in loops, etc.
    coord_length, num_coords = important_atom_positions.shape
    total_atoms = len(all_atom_positions)
    print(f"Length of all_atom_positions:    {total_atoms}")
    print(f"Length of important_atom_positions: {coord_length}")
    
    # NEAREST NEIGHBOR ALGORITHM:
    # First, generate Neighborhoods by partitioning the list of positions into groups of size neighborhood_partition_size,
    # where any remainder at the end is grouped together in a Neighborhood which is smaller than the rest (the length of 
    # the list of atom positions will generally not be nicely divisible into equal groups so the final group will be smaller).
    neighborhood_partition_size = 100
    neighborhood_list = []
    
    head = 0
    while(head < coord_length):
        if(head+neighborhood_partition_size > coord_length):
            neighborhood_list.append(Neighborhood(important_atom_positions[head:coord_length], rnn))
            head = coord_length
        else:
            neighborhood_list.append(Neighborhood(important_atom_positions[head:head+neighborhood_partition_size], rnn))
            head += neighborhood_partition_size

    num_neighborhoods = len(neighborhood_list)
    print(f"Number of neighborhoods: {num_neighborhoods}")
    
    for i in range(len(neighborhood_list)):
        neighborhood_list[i].generate_nearest_neighbor_vecs()
    
    
    # Calculate QL, a bond order parameter which is averaged over all nearest neighbor pairs in a neighborhood 
    # (then averaged over all neighborhoods)
    QL = 0.0
    QLM_average = 0.0
    total_nn_pairs = 0
    
    for m in range (-L, L+1):
        QLM_average = 0.0
        for i in range(num_neighborhoods):
            QLM_average += neighborhood_list[i].calc_QLM(m, L)
            total_nn_pairs += len(neighborhood_list[i].get_nearest_neighbor_vecs())
        QLM_average /= total_nn_pairs
        QL += abs(QLM_average)**2

    QL = np.sqrt(QL*((4*np.pi)/(2*L + 1)))
    QL_list.append(QL)
    print(f"Q{L} value for this frame: {QL_list[iframe]:.4f}\n")

end_time = time.process_time()
elapsed_time = end_time - start_time

file_object.close()

#print("List of lines/atoms read:\n")
#print(atom_list)
#print()
#print("List of coordinates for each atom:\n")
#print(important_atom_positions)
#print()
#print("List of nearest neighbors (by list index) for each atom:\n")
#print(nn_list)
print()
#print(f"Calculated value of Q{L}: {QL}\n")
print()
print(f"Elapsed time was {elapsed_time} seconds.")

# Graphing the data
#fig, ax = plt.subplots()
#L_string = "Q" + str(L)
#_title = "Value of " + L_string + " Frame-by-Frame"
#ax.plot(frame_list, QL_list, linestyle='dotted', markersize=8)
#ax.scatter(frame_list, QL_list, color='r')
#ax.set(title=_title, xlabel='Frame', ylabel=L_string)
#plt.show()

# Writing the data to a text file:
with open(f'./Q_output_{file_name}.txt', 'w') as f:
    f.write(f'Bond Order Parameters for {file_name}\n')
    f.write(f'Frame              Q{L}\n')
    for i in range(len(QL_list)):
        f.write(f'{frame_list[i]}               {QL_list[i]:.4f}\n')


# Pseudocode for math:
# select an atom
# define a vector r from center of that atom to another
# check if nearest_neighbor (within rnn); if so, move onto math / if not, skip to next atom
# math: use vector r from atom to neighbor to calc. angles relative to coordinate frame
# plug in angles to spherical harmonic equation to get Qlm
# average Qlm over all bonds (nearest-neighbor pairs) in sample
# plug average into Eq. 1.3 from Steinhardt to calculate Ql
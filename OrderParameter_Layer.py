# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 18:03:01 2021

@author: Armand Parada Jackson
"""

# Program to perform order parameter calculations on input coordinates

#from mpmath import * 
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from math import sqrt
import ase # just import what you need
from ase import io
from pathlib import Path

#input_path = str(sys.argv[1])
#L = int(sys.argv[2])

directory = "C:/Users/arman/Desktop/College/UNT/Research/MateriaLab/Programs/"

#file_name = input("Enter file name: ")
file_name = "hydrate.pdb"
input_path = Path(directory)/file_name

if not input_path.exists():
    print(f"The path {input_path} does not exist!")
    sys.exit(1)
    
#L = int(input("Enter L value:"))
L = 6

start_time = time.process_time()

r0 = 2.75 # diameter of water molecule in angstroms; in general,
          # first peak in radial distribution (Steinhardt)
rnn = 1.2*r0 # nearest neighbor threshold distance, in factor of r0
rnn2 = rnn**2

file_object = open(input_path, "r")
#file_string = file_object.read()
#framesTotal = file_string.count('frame')
num_frames = 10

#file_object.seek(0,0)

QL_list = []

for iframe in range(num_frames):
    
    print(f"Current frame is: {iframe}")
    
    # stores 2D numpy array of atoms in fileData
    fileData = ase.io.read(input_path,index=iframe,format='proteindatabank')
    symbolData = np.array(fileData.get_chemical_symbols())
    posData: np.ndarray = fileData.get_positions()
    posData = posData[(symbolData != 'H')] # filter out hydrogen atoms from posData
    coord_list = posData[(posData[:,2] >= 39) & (posData[:,2] <= 58)]# * only store atoms with z between 39 and 58 in coord_list
    print(f"Length of posData: {len(posData)}")
    print(posData)
    #sys.exit(1)
    # Filtering works
    
    coord_length, num_coords = coord_list.shape
    total_atoms = len(posData)
    
    dx = 0.0 # variables storing coord. differences for nearest neighbors
    dy = 0.0
    dz = 0.0
    
    # nn_list stores sublists of NN vectors, FROM an atom TO its nearest neighbor
    nn_list = [[[]] for r in range(coord_length)]
    dx = 0.0
    dy = 0.0
    dz = 0.0
    percentProgress = 0.0
    
    # Find nearest neighbors
    for i in range(coord_length):
        for j in range(total_atoms):
                dx = posData[j][0] - coord_list[i][0]
                dy = posData[j][1] - coord_list[i][1]
                dz = posData[j][2] - coord_list[i][2]
                dr2 = dx**2 + dy**2 + dz**2
                if(dr2 <= rnn2 and dr2 != 0):
                    nn_list[i].append([dx, dy, dz])
                    print(f"Process: {percentProgress} / 100 complete")
    nn_list = np.array(nn_list)
    print(nn_list)
    nn_list = sqrt(nn_list)
    # ******** YOU ARE HERE, NN SLOW ***********
    
    
    # calculates QLM given a vector r and quantum numbers m & L
    def calc_QLM(r, m, L):
        r_mag = sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        if(r[0] < 0.001):
            r[0] = 0.001
        theta = np.arctan(r[1]/r[0]) # azimuthal angle
        phi = np.arccos(r[2]/r_mag) # polar/colatitudinal angle
        
        return special.sph_harm(m, L, theta, phi)
    
    # Calculate QL
    QLM = []
    QL = 0.0
    
    for m in range (-L, L+1):
        for i in range(len(nn_list)):
            for j in range(len(nn_list[i])):
                if(nn_list[i][1] != []): # get r vector from atom i to NN atom j
                    QLM.append(calc_QLM(nn_list[i][j], m, L))
        QLM_avg = np.sum(QLM)/len(QLM)
        QL += abs(QLM_avg)**2
    
    QL *= (4*np.pi)/(2*L + 1)
    QL = np.sqrt(QL)
    QL_list.append(QL)
    print(f"Q{L} value for this frame: {QL_list[iframe]}\n")

end_time = time.process_time()
elapsed_time = end_time - start_time

file_object.close()

#print("List of lines/atoms read:\n")
#print(atom_list)
#print()
#print("List of coordinates for each atom:\n")
#print(coord_list)
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
ax.plot(frameList, QL_list, linestyle='dotted', markersize=8)
ax.scatter(frameList, QL_list, color='r')
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
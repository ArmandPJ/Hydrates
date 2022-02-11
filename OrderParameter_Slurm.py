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
import scipy as sci
from scipy import special
from math import sqrt
import ase
from ase import io
from tkinter import *
from tkinter import ttk


input_path = str(sys.argv[1])
L = int(sys.argv[2])

if not os.path.exists(input_path):
    print(f"The path {input_path} does not exist!")
    sys.exit(1)

start_time = time.process_time()

file_object = open(input_path, "r")

r0 = 2.75 # diameter of water molecule in angstroms; in general,
          # first peak in radial distribution (Steinhardt)
rnn = 1.2*r0 # nearest neighbor threshold distance, in factor of r0

file_string = file_object.read()
framesTotal = file_string.count('frame')
frames = 10
frameList = [f for f in range(frames)]

file_object.seek(0,0)

QL_list = []

for frame in range(frames):
    # stores 2D numpy array of coordinates of atoms in coord_list
    coord_listMain = ase.io.read(input_path,index=frame,format='proteindatabank').get_positions()
    coord_list = coord_listMain
    #print(type(coord_list))
    #print(coord_list)
    coord_length, num_coords = coord_list.shape
    #print(f"Coord array: {coord_length} x {num_coords}")
    
    dx = 0.0 # variables storing coord. differences for nearest neighbors
    dy = 0.0
    dz = 0.0
    
    # nn_list stores sublists of nearest neighbors for all atoms
    nn_list = [[] for r in range(coord_length)]
    
    for i in range(coord_length-1):
        for j in range(coord_length-1):
            if(i != j):
                dx = coord_list[i][0] - coord_list[j][0]
                dy = coord_list[i][1] - coord_list[j][1]
                dz = coord_list[i][2] - coord_list[j][2]
                dr = sqrt(dx**2 + dy**2 + dz**2)
                if(dr <= rnn):
                    if(j in nn_list[i]) != True:
                        nn_list[i].append(j)
                        # i is atom index, j is index of a NN to that atom
                    if(i in nn_list[j]) != True:
                        nn_list[j].append(i) # both atom i and j are NN of each other
    # Nearest neighbor generation WORKS!
    
    # Convert nn_list from nested list to numpy array
    #nn_list = np.array([np.array(l) for l in nn_list])
    #nn_length = len(nn_list)
    
    # calculates QLM given a vector r and quantum numbers m & L
    def calc_QLM(r, m, L):
        r_mag = sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        if(r[0] < 0.001):
            r[0] = 0.001
        theta = np.arctan(r[1]/r[0]) # azimuthal angle
        phi = np.arccos(r[2]/r_mag) # polar/colatitudinal angle
        
        return sci.special.sph_harm(m, L, theta, phi)
    
    # Calculate Q6 (L = 6)
    QLM = []
    QL = 0.0
    
    for m in range (-L, L+1):
        for i in range(len(nn_list)):
            for j in range(len(nn_list[i])):
                if(nn_list[i] != []): # find r vector from atom i to NN atom j
                    r_vector = [coord_list[i][c] - coord_list[nn_list[i][j]][c] for c in range(3)]
                    QLM.append(calc_QLM(r_vector, m, L))
        QLM_avg = np.sum(QLM)/len(QLM)
        QL += abs(QLM_avg)**2
    
    QL *= (4*np.pi)/(2*L + 1)
    QL = np.sqrt(QL)
    QL_list.append(QL)
    print(f"Q{L} value for this frame: {QL_list[frame]}\n")

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
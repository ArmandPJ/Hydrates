# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 18:03:01 2021

@author: Armand Parada Jackson
"""

# Program to perform order parameter calculations on input coordinates

#from mpmath import * 
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from scipy import special
from math import sqrt

while(True):
    file_name = input("Enter file name: ")
    if(os.path.isfile("./" + file_name)):
        print(f"The file {file_name} exists!\n")
        L = int(input("Enter l value: "))
        print()
        break
    else:
        print(f"The file {file_name} does NOT exist! Try again.\n")

start_time = time.process_time()

file_object = open(file_name, "r")

r0 = 2.75 # diameter of water molecule in angstroms; in general,
          # first peak in radial distribution (Steinhardt)
rnn = 1.2*r0 # nearest neighbor threshold distance, in factor of r0


atom_list = []
coord_list = []

# 23,055 lines from REMARK: Frame line to END line, in "hydrate.pdb"
for i in range(5000): # how many lines to read
    cur = file_object.readline(); # cur stores the current or most recent line
    if(cur.find('ATOM') != -1): # if an ATOM line, store in atom_list
        line_data = cur.split()
        atom_list.append(line_data)

for i in range(len(atom_list)):
    temp_coord_list = [] # stores xyz coords as nested lists in coord_list
    for j in range(5,8): # column 5 through 7 in split text file give xyz coords
        temp_coord_list.append(float(atom_list[i][j]))
    coord_list.append(temp_coord_list)

dx = 0.0 # variables storing coord. differences for nearest neighbors
dy = 0.0
dz = 0.0

# nn_list stores sublists of nearest neighbors for all atoms
nn_list =[[] for i in range(len(atom_list))]

for i in range(len(coord_list)-1):
    for j in range(len(coord_list)-1):
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

for m in range (-L, L):
    for i in range(len(nn_list)):
        for j in range(len(nn_list[i])):
            if(nn_list[i] != []): # find r vector from atom i to NN atom j
                r_vector = [coord_list[i][c] - coord_list[nn_list[i][j]][c] for c in range(3)]
                QLM.append(calc_QLM(r_vector, m, L))
    QLM_avg = np.sum(QLM)/len(QLM)
    QL += abs(QLM_avg)**2

QL *= (4*np.pi)/(2*L + 1)
QL = np.sqrt(QL)

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
print(f"Calculated value of Q{L}:\n")
print(QL)
print()
print(f"Elapsed time was {elapsed_time} seconds.")

# Pseudocode for math:
# select a molecule
# define a vector r from center of molecule to center of another
# check if nearest_neighbor; if so, move onto math / if not, skip to next neighbor
# math: use vector r from molecule to neighbor to calc. angles relative to coordinate frame
# plug in angles to spherical harmonic equation to get Qlm
# average Qlm over all bonds in sample
# plug average into Eq. 1.3 from Steinhardt to calculate Ql
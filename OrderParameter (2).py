# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 18:03:01 2021

@author: Armand Parada Jackson
"""

# Program to perform order parameter calculations on input coordinates

from mpmath import * 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci

file_object = open("hydrate.pdb", "r") # replace "filename" with input file name

x = 0.0
y = 0.0
z = 0.0

theta = 0.0
phi = 0.0

r0 = 2.75 # diameter of water molecule in angstroms; in general,
          # first peak in radial distribution (Steinhardt)
rnn = 1.2*r0 # nearest neighbor threshold distance, in factor of r0

atom_list = []
coord_list = []

for i in range(10): # how many lines to read
    cur = file_object.readline(); # cur stores the current or most recent line
    if(cur.find('ATOM') != -1): # if an ATOM line, store in atom_list
        line_data = cur.split()
        atom_list.append(line_data)

for i in range(len(atom_list)):
    temp_coord_list = [] # stores xyz coords as nested lists in coord_list
    for j in range(5,8): # column 5 through 7 in text file give xyz coords
        temp_coord_list.append(atom_list[i][j])
    coord_list.append(temp_coord_list)

dx = 0.0 # temp. variables storing coord. differences for nearest neighbors
dy = 0.0
dz = 0.0

# nn_list stores sublists of nearest neighbors for all atoms
nn_list =[[] for i in range(len(atom_list))]

for i in range(len(coord_list)-1):
    for j in range(len(coord_list)-1):
        if(i != j):
            dx = float(coord_list[i][0]) - float(coord_list[j][0])
            dy = float(coord_list[i][1]) - float(coord_list[j][1])
            dz = float(coord_list[i][2]) - float(coord_list[j][2])
            dr = (dx**2 + dy**2 + dz**2)**0.5
            if(dr <= rnn):
                if(j in nn_list[i]) != True:
                    nn_list[i].append(j)  
                    # i is atom index, j is index of a NN to that atom
                if(i in nn_list[j]) != True:
                    nn_list[j].append(i) # both atom i and j are NN of each other
# Nearest neighbor generation WORKS!

file_object.close()

# finds a specific pattern of elements in mylist
def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append(pattern)
    return matches

print("List of lines/atoms read:\n")
print(atom_list)
print()
print("List of coordinates for each atom:\n")
print(coord_list)
print()
print("List of nearest neighbors (by index) for each atom:\n")
print(nn_list)

# Pseudocode for math:
# select a molecule
# define a vector r from center of molecule to center of another
# check if nearest_neighbor; if so, move onto math / if not, skip to next neighbor
# math: use vector r from molecule to neighbor to calc. angles relative to coordinate frame
# plug in angles to spherical harmonic equation to get Qlm
# average Qlm over all bonds in sample
# plug average into Eq. 1.3 from Steinhardt to calculate Ql
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

def filter_atom_arrays(_all_atoms, _symbols):
    """ Filters an array via various techniques
    
    Parameters:
    _all_atoms (ndarray): an array of atom positions (vectors)

    Returns:
    surface_atoms (ndarray): an array of surface atoms within a particular range of z
    all_atoms (ndarray): an array of all atoms within certain filtered conditions

    """
    all_atoms = _all_atoms
    all_atoms = all_atoms[(_symbols != 'H')] # filter out hydrogen atoms from all_atoms
    all_atoms = all_atoms[all_atoms[:,2].argsort()] # sort the array by the z coordinate

    z_min = 39 # narrow down the array to a layer of atoms between z = 39 and z = 60
    z_max = 60
    surface_width_in_z = z_max - z_min

    # filter out any atoms that are more than 1 surface width away (in z) from surface of interest
    all_atoms = all_atoms[(all_atoms[:,2] >= z_min - surface_width_in_z)
                            & (all_atoms[:,2] <= surface_width_in_z + z_max)]
    
    # store surface atoms with z between 39 and 58 in surface_atom_positions
    surface_atoms = all_atoms[(all_atoms[:,2] >= z_min) & (all_atoms[:,2] <= z_max)]

    return surface_atoms, all_atoms

def normalize_by_column(arr, coord_widths):
    """ Normalizes an array by column
    
    Parameters:
    arr (ndarray): array to be normalized

    Returns:
    _array (ndarray): normalized copy of arr
    
    """
    _array = arr
    for col in range(3):
        _array[:, col] = (_array[:,col] - np.min(_array[:,col])) / coord_widths[col]

    return _array

def denormalize_by_column(arr, coord_widths):
    """ Undoes normalization on a given array
    
    Parameters:
    arr (ndarray): the array to be denormalized

    Returns:
    _array (ndarray): a denormalized copy of arr

    """
    _arr = arr
    for col in range(3):
        _arr[:, col] = _arr[:,col] * coord_widths[col] + np.min(_arr[:, col])

    return _arr

def scale_and_adjust_periodicity(_diff_vectors, coord_widths):
    """ Normalizes a given array, adjusts for periodic boundaries, then denormalizes the array
    
    Parameters:
    _diff_vectors (ndarray): an array to be scaled and adjusted for periodicity

    Returns:
    difference_vectors (ndarray): a copy of the original array, adjust for periodic boundaries
    
    """
    difference_vectors = _diff_vectors
    difference_vectors = normalize_by_column(difference_vectors, coord_widths)
    for col in range(2): # adjust x and y to the mirror image of the hydrate (if needed) to check periodic neighbors
        difference_vectors[:, col] = difference_vectors[:,col] - np.around(difference_vectors[:,col], 0)
    difference_vectors = denormalize_by_column(difference_vectors, coord_widths)
    
    return difference_vectors

def generate_nearest_neighbors(surface_atom_positions, all_atom_positions, iframe, num_frames, coord_widths, rnn):
    """ Generates a list of nearest neighbor vectors

    Parameters:
    surface_atom_positions (ndarray): array of position vectors of atoms at the hydrate surface
    all_atom_positions (ndarray): array of position vectors of all atoms in the hydrate (filtered)
    iframe (int): the current frame of the .pdb file
    num_frames (int): the total number of frames in the .pdb file
    coord_widths (ndarray): an array containing the 3 widths of the hydrate in x, y, and z
    rnn (float): nearest neighbor threshold distance
    
    Returns:
    nearest_neighbor_vecs (ndarray): an array of nearest neighbor vectors
    num_neighbor_vecs (int): the number of nearest neighbor vectors 
    
    """
    potential_neighbor_vectors = np.empty((0,3))

    for current_atom_index in range(len(surface_atom_positions)):

        this_atom = surface_atom_positions[current_atom_index]

        # Narrow the search to atoms whose z component alone is within rnn distance
        atoms_to_check = all_atom_positions[(abs(all_atom_positions[:,2] - this_atom[2]) <= rnn)]

        diff_vectors = atoms_to_check - this_atom
        diff_vectors = scale_and_adjust_periodicity(diff_vectors, coord_widths)
        #TODO Scale and adjust for periodicity

        potential_neighbor_vectors = np.append(potential_neighbor_vectors, diff_vectors, axis=0)

        clear()
        print(f"Current Frame is: {iframe+1}/{num_frames}\n")
        print(f"{current_atom_index+1}/{len(surface_atom_positions)} surface atoms scanned.")

    nearest_neighbor_vecs = potential_neighbor_vectors[(np.linalg.norm(potential_neighbor_vectors[:], axis=1) <= rnn)
                                                            & (np.linalg.norm(potential_neighbor_vectors[:], axis=1) != 0)]
    num_neighbor_vecs = len(nearest_neighbor_vecs)
    print(f"Nearest neighbor vectors length: {num_neighbor_vecs}")
    print(nearest_neighbor_vecs)
    
    return nearest_neighbor_vecs, num_neighbor_vecs

def calc_QLM(r_vector, _m, _L):
    """ Calculates QLM of a specific vector
    
    Parameters:
    r_vector (ndarray): a 3D vector
    _m (int): the value of quantum number m
    _L (int): the value of quantum number L

    Returns:
    special.sph_harm(_m, _L, theta, phi) (complex float): the spherical harmonic associated with _m, _L, theta and phi
    
    """
    r_mag = np.linalg.norm(r_vector)
    if(r_vector[0] == 0.0):
        if(r_vector[1] > 0): theta = np.pi/2
        elif(r_vector[1] < 0): theta = 3*(np.pi/2)
    else: theta = np.arctan(r_vector[1]/r_vector[0]) # azimuthal angle
    if(theta < 0): theta += 2*np.pi # theta must be between 0 and 2*pi

    phi = np.arccos(r_vector[2]/r_mag) # polar/colatitudinal angle
    if(phi < 0): phi += np.pi # phi must be between 0 and pi
    
    return special.sph_harm(_m, _L, theta, phi)

def write_to_file(L, file_name, QL_list, frame_list):
    """ 
    Writes the values of QL of each frame to a text file

    """
    with open(f'./output_files/Q{L}_output_{file_name}.txt', 'w') as f:
        f.write(f'Bond Order Parameters for {file_name}\n')
        f.write(f'Frame          Q{L}\n')
        for i in range(len(QL_list)):
            f.write(f'{frame_list[i]}{QL_list[i]:20.4f}\n')

def main():
    """"""
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
        
        # Filter function
        surface_atom_positions, all_atom_positions = filter_atom_arrays(all_atom_positions, symbol_data)
        
        # Get useful lengths and dimensions of arrays for printing, later use in loops, etc.
        num_atoms_in_surface, num_coords = surface_atom_positions.shape
        total_atoms = len(all_atom_positions)
        print(f"Length of all_atom_positions:    {total_atoms}")
        print(f"Length of surface_atom_positions:  {num_atoms_in_surface}\n")

        # Get the width of the hydrate in x, y, z
        maxima = np.max(all_atom_positions, 0)
        minima = np.min(all_atom_positions, 0)
        # maxima and minima are arrays with elements of the maximum or minimum values of each column in all_atom_positions
        coord_widths = maxima - minima

        # Generate nearest neighbor vectors
        nearest_neighbor_vectors, num_neighbor_vectors = generate_nearest_neighbors(surface_atom_positions, all_atom_positions,
                                                                                    iframe, num_frames, coord_widths, rnn)

        # Calculate QL, a bond order parameter which is averaged over all nearest neighbor vectors
        QLM = []
        QL = 0.0
        
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

    write_to_file(L, file_name, QL_list, frame_list)

if __name__ == "__main__":
    main()


# Pseudocode for math:
# select an atom
# define a vector r from center of that atom to another
# check if nearest_neighbor (within rnn); if so, move onto math / if not, skip to next atom
# math: use vector r from atom to neighbor to calc. angles between nearest neighbors
# plug in angles to spherical harmonic equation to get Qlm
# average Qlm over all bonds (nearest-neighbor pairs) in sample
# plug average into Eq. 1.3 from Steinhardt to calculate Ql
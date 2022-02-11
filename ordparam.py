# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:05:32 2022

@author: Armand Parada Jackson
"""

import sys
import numpy as np

class Neighborhood:
    def __init__(self, arr, nn_distance):
        self.atoms_array = arr
        self.number_residents = len(arr)
        self.nn_vectors = np.empty([0, 3])
        self.number_neighbors = 0
        self.nn_magnitudes = np.empty([0])
        self.thresh = nn_distance
        self.nn_generated = False
        
        
    def __str__(self):
        return f"This Neighborhood ({self.number_residents} residents):\n" + str(self.atoms_array) + "\n" + "\n"
        
    def getAtoms(self):
        # Return an array of all of the atoms' positions in this neighborhood.
        return self.atoms_array
    
    def generate_nearest_neighbor_vecs(self):
        # Generate a numpy array of vectors for nearest neighbors in the neighborhood.
        # First, create an array of all of the relative position vectors for every possible pair of atoms in the neighborhood.
        for i in range(self.number_residents):
            for j in range(self.number_residents):
                if(i != j):
                    # Get the vector FROM atom i TO atom j, and append that vector the numpy array nn_vectors.
                    # Also, keep track of the magnitudes of each vector in a separate list, nn_magnitudes.
                    vec = self.atoms_array[j] - self.atoms_array[i]
                    self.nn_vectors = np.append(self.nn_vectors, [vec], axis=0)
                    self.nn_magnitudes = np.append(self.nn_magnitudes, np.linalg.norm(vec))
        
        # Then, filter out all vectors which are NOT within threshold distance, thresh.
        self.nn_vectors = self.nn_vectors[(self.nn_magnitudes[:] <= self.thresh)]
        self.nn_magnitudes = self.nn_magnitudes[(self.nn_magnitudes[:] <= self.thresh)]
        
        # Finally, return the array nn_vectors.
        self.nn_generated = True
        self.number_neighbors = len(self.nn_vectors)
    
    def get_nearest_neighbor_vecs(self):
        # If the nearest neighbor list is already generated, then simply return it. If not, generate it and return it.
        if(self.nn_generated):
            return self.nn_vectors
        else:
            self.generate_nearest_neighbor_vecs()
            return self.nn_vectors

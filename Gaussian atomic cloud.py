# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:24:48 2024

@author: kousi
"""

import numpy as np
import matplotlib.pyplot as plt


def create_atomic_cloud(center_density, sigma):
    """
    Create a model of an atomic cloud with a Gaussian density distribution.

    Args:
    - center_density: density of atoms at the center (units: 10^18 m^-3)
    - sigma: standard deviation of the density profile (10^-6 m)

    Returns:
    - atoms: array containing coordinates of atoms in the cloud
    """
    # Generate random coordinates
    max_coordinate = 4 * sigma  # 4 sigma to ensure enough coverage
    num_atoms = num_atoms = center_density * (2*np.pi*sigma**2)**1.5  # found by doing the triple integral
    num_atoms = int(num_atoms)
    x = np.random.uniform(-max_coordinate, max_coordinate, size=num_atoms)
    y = np.random.uniform(-max_coordinate, max_coordinate, size=num_atoms)
    z = np.random.uniform(-max_coordinate, max_coordinate, size=num_atoms)

    # distances of each atom from the center
    distances = np.sqrt(x**2 + y**2 + z**2)

    # probability density function (PDF) of the Gaussian distribution
    pdf = center_density * np.exp(-0.5 * (distances / sigma)**2)

    # Normalized the PDF to ensure that the sum equals the total number of atoms
    pdf /= np.sum(pdf)

    # Sample positions based on the PDF to generate the atoms
    sampled_indices = np.random.choice(num_atoms, size=num_atoms, p=pdf)
    atoms = np.column_stack((x[sampled_indices], y[sampled_indices], z[sampled_indices]))

    return atoms

# Parameters for the atomic cloud
center_density = 5  # density of atoms at the center 
sigma = 5  # standard deviation of the density profile


# Create the atomic cloud
atomic_cloud = create_atomic_cloud(center_density, sigma)

# Plot the atomic cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(atomic_cloud[:, 0], atomic_cloud[:, 1], atomic_cloud[:, 2], c='b', marker='.')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Atomic Cloud with Gaussian Density Distribution')
plt.show()














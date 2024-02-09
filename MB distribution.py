# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:07:56 2024

@author: kousi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

def create_atomic_cloud(center_density, sigma, mass, temperature):
    """
    Create a model of an atomic cloud with a Gaussian density distribution and Maxwell-Boltzmann velocity distribution.

    Args:
    - center_density: density of atoms at the center (units: 10^18 m^-3)
    - sigma: standard deviation of the density profile (10^-6 m)
    - mass: mass of the atoms (kg)
    - temperature: temperature of the atomic cloud (K)

    Returns:
    - atoms: array containing coordinates and velocities of atoms in the cloud
    """
    # Generate random coordinates
    max_coordinate = 4 * sigma  # 4 sigma to ensure enough coverage
    num_atoms = center_density * (2*np.pi*sigma**2)**1.5  # number of atoms based on density
    num_atoms = int(num_atoms)
    x = np.random.uniform(-max_coordinate, max_coordinate, size=num_atoms)
    y = np.random.uniform(-max_coordinate, max_coordinate, size=num_atoms)
    z = np.random.uniform(-max_coordinate, max_coordinate, size=num_atoms)

    # distances of each atom from the center
    distances = np.sqrt(x**2 + y**2 + z**2)

    # probability density function (PDF) of the Gaussian distribution
    pdf = center_density * np.exp(-0.5 * (distances / sigma)**2)

    # Normalize the PDF to ensure that the sum equals the total number of atoms
    pdf /= np.sum(pdf)

    # Sample positions based on the PDF to generate the atoms
    sampled_indices = np.random.choice(num_atoms, size=num_atoms, p=pdf)
    atoms = np.column_stack((x[sampled_indices], y[sampled_indices], z[sampled_indices]))

    # Generate random velocities based on Maxwell-Boltzmann distribution
    k_B = constants.k  # Boltzmann constant (m^2 kg s^-2 K^-1)
    mean_speed = np.sqrt((8 * k_B * temperature) / (np.pi * mass))
    velocities = np.random.normal(0, mean_speed, size=(num_atoms, 3))

    # Combine coordinates and velocities into one array
    atoms_with_velocities = np.hstack((atoms, velocities))

    return atoms_with_velocities

# Parameters for the atomic cloud
center_density = 5  # density of atoms at the center 
sigma = 5  # standard deviation of the density profile
mass = 86.9 * constants.atomic_mass  # mass of the Rb87 atoms (kg)
temperature = 1e-6  # temperature of the atomic cloud (K)

# Create the atomic cloud with velocities
atomic_cloud_with_velocities = create_atomic_cloud(center_density, sigma, mass, temperature)


def plot_atomic_cloud_velocity(atomic_cloud):
    """
    Plot the absolute value of the velocity of each atom in the atomic cloud.

    Args:
    - atomic_cloud: array containing coordinates and velocities of atoms in the cloud
    """
    # Extract velocity components
    velocities = atomic_cloud[:, 3:]

    # Calculate absolute value of velocity vector
    absolute_velocity = np.linalg.norm(velocities, axis=1)

    # Plot the absolute value of velocity
    plt.hist(absolute_velocity, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Absolute Velocity (m/s)')
    plt.ylabel('Frequency')
    plt.title('Absolute Velocity Distribution of Atomic Cloud')
    plt.show()

plot_atomic_cloud_velocity(atomic_cloud_with_velocities)

def plot_atomic_cloud_real_space(atomic_cloud):
    """
    Plot the atoms in real space (without considering velocities).

    Args:
    - atomic_cloud: array containing coordinates and velocities of atoms in the cloud
    """
    # Extract coordinates
    coordinates = atomic_cloud[:, :3]

    # Plot the atoms in real space
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='b', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Atomic Cloud in Real Space')
    plt.show()

plot_atomic_cloud_real_space(atomic_cloud_with_velocities)


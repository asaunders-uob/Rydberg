# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:55:25 2024

@author: kousi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy import constants
from scipy.integrate import quad
from scipy.interpolate import interp1d

hbar = constants.hbar
m = 86.9 * constants.atomic_mass  # mass of the Rb87 atoms (kg)
k = constants.k
T = 1e-7  # temperature of the atomic cloud (K)
beta = 1/(k*T)

omega = 1 * 1e3


N = 2000
r = np.linspace(-10, 10, N)*1e-6

mu = 0
mu_bosons = -k*T/N
E_F = hbar * omega * (6*N)**(1/3)

de_Broglie = np.sqrt((2*np.pi*hbar**2)/(m*k*T))

def harmonic_potential(omega, mass, r):
    V = 0.5 * mass * (omega**2 * r**2)
    return V

V = harmonic_potential(omega, m, r)


def integrand(q, n, z):
    return (q**(n-1))/(np.exp(q) / z - 1)

def Li(n, z):
    integrals = np.array([])
    for i in z:
        integral = quad(integrand, 0, np.inf, args=(n,i))[0]
        integrals = np.append(integrals, integral)
    return integrals / gamma(n)

def density_distribution(r, deg=0, mu=0):
    n=(-1)**deg * (1/de_Broglie**3)*Li(1.5, (-1)**deg * np.exp(beta*(mu-harmonic_potential(omega, m, r))))
    return n


def MB(r):
    #N = N_th(omega, T, mu_bosons, deg=1)
    sigma = np.sqrt(2*k*T/(m*omega**2))
    return N/(np.pi**1.5 * sigma**3) * np.exp(-r**2 / sigma**2)

bosons = density_distribution(r, deg=0, mu=0)
fermions = density_distribution(r, deg=1, mu=0)
mb = MB(r)

area_b = np.trapz(bosons, r, dx=0.1)
area_f = np.trapz(fermions, r, dx=0.1)
area_mb = np.trapz(mb, r, dx=0.1)

plt.plot(r*1e6, bosons/area_b, label='Bosons')
plt.plot(r*1e6, fermions/area_f, label='Fermions')
plt.plot(r*1e6, mb/area_mb, label='Maxwell Boltzman')
plt.legend()
plt.xlabel('r (μm)')
plt.ylabel('PDF')
plt.savefig('PDF.png', bbox_inches='tight')

psi_0 = 1/np.sqrt(N) * np.ones((N, 1))
#print(psi_0)

psi_t = np.transpose(psi_0)

x = np.linspace(-10, 10, N)


t = np.linspace(0, 100, 150)*1e-6

values_b = np.array([])
for i in t:

    psi_f = 1/np.sqrt(N) * np.exp(-bosons/area_b * i * 1j)
    
    psi_f = np.reshape(psi_f, (1,N))
    
    ans = np.dot(psi_f, psi_0)
    values_b = np.append(values_b, np.abs(ans)**2)
    
values_f = np.array([])
for i in t:

    psi_f = 1/np.sqrt(N) * np.exp(-fermions/area_f * i * 1j)
    
    psi_f = np.reshape(psi_f, (1,N))
    
    ans = np.dot(psi_f, psi_0)
    values_f = np.append(values_f, np.abs(ans)**2)
    
values_mb = np.array([])
for i in t:

    psi_f = 1/np.sqrt(N) * np.exp(-mb/area_mb * i * 1j)
    
    psi_f = np.reshape(psi_f, (1,N))
    
    ans = np.dot(psi_f, psi_0)
    values_mb = np.append(values_mb, np.abs(ans)**2)


plt.figure()
plt.plot(t*1e6, values_b, label='Bosons')
plt.plot(t*1e6, values_f, label='Fermions')
plt.plot(t*1e6, values_mb, label='Maxwell Boltzman')
plt.legend()
plt.xlabel('Time (μs)')
plt.ylabel('Modululs of Expectation Value')
plt.savefig('expectation.png', bbox_inches='tight')
'''
plt.figure()
# Calculate the cumulative distribution function (CDF)
def calculate_cdf(pdf, x_values):
    cdf_values = [quad(pdf, -10*1e-6, x)[0] for x in x_values]
    cdf_values = np.array(cdf_values)  # Convert to NumPy array
    cdf_values /= cdf_values[-1]  # Normalize to [0, 1]
    return cdf_values

# Generate random samples from a uniform distribution
num_samples = N
uniform_samples = np.random.uniform(0, 1, num_samples)

# Calculate the inverse of the CDF using interpolation
x_values = r
cdf_values = calculate_cdf(bosons, x_values)
inverse_cdf = interp1d(cdf_values, x_values, kind='linear', fill_value='extrapolate')

# Transform uniform samples to samples from the custom distribution
generated_samples = inverse_cdf(uniform_samples)

#plt.hist(generated_samples, bins=50, range=(-10*1e-6, 10*1e-6), density=True)

# Plot the histogram of generated data
hist_counts, bin_edges = np.histogram(generated_samples, bins=50)

# Calculate the bin heights
bin_widths = np.diff(bin_edges)
bin_heights = hist_counts / bin_widths


# Create a list containing all the heights of the data points
heights_list = []
for count, height in zip(hist_counts, bin_heights):
    heights_list.extend([height] * int(count))

# Print the first few elements of the heights list
#print("Heights list:", heights_list[:20])
print(np.shape(heights_list))

plt.figure()
heights = np.array(heights_list)
values_trial = np.array([])
for i in t:

    psi_f = 1/np.sqrt(N) * np.exp(-heights/1e3 * i * 1j)
    
    psi_f = np.reshape(psi_f, (1,N))
    
    ans = np.dot(psi_f, psi_0)
    values_trial = np.append(values_trial, np.abs(ans)**2)
    
plt.plot(t*1e6, values_trial)
'''







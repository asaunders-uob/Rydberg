# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:07:52 2024

@author: kousi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy import constants
from scipy.special import diric
from scipy.integrate import quad

hbar = constants.hbar
m = 86.9 * constants.atomic_mass  # mass of the Rb87 atoms (kg)
k = constants.k
T = 1e-7  # temperature of the atomic cloud (K)
beta = 1/(k*T)

omega = 1 * 1e3
points = 1000
r = np.linspace(0, 10, points)*1e-6
N = 2*1e4
mu = 0
mu_bosons = -k*T/N
E_F = hbar * omega * (6*N)**(1/3)

de_Broglie = np.sqrt((2*np.pi*hbar**2)/(m*k*T))

def occupation_number(E, mu, beta, deg=0):
    '''
    Parameters
    ----------
    E : Particle Energy
    mu : Chemical Potential
    beta : 1/kT
    deg : bosons: 0, fermions: 1
        DESCRIPTION. The default is 0.

    Returns
    -------
    occupation number
    '''
    return 1/(np.exp(beta*(E-mu))+(-1)**deg)

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

plt.plot(r, density_distribution(r, deg=0, mu=0))
plt.plot(r, density_distribution(r, deg=1, mu=0))

def N_th(omega, T, mu, deg=0):
    return (-1)**deg * ((k*T)/(hbar*omega))**3 * Li(3, [(-1)**deg * np.exp(mu/(k*T))])

print(N_th(omega, T, mu_bosons, deg=0))
print(N_th(omega, T, mu_bosons, deg=1))

def n_B(r):
    return 1/de_Broglie**3 * Li(1.5, np.exp(-beta*harmonic_potential(omega, m, r)))

#plt.plot(r, n_B(r))

def n_F(r, mu=0):  # how? -V(r) is negative and should be raised to 3/2?
    mu = E_F - harmonic_potential(omega, m, r)
    n = (1/(6*np.pi**2))*(2*m/hbar**2)**1.5 * (mu-harmonic_potential(omega, m, r))**1.5
    return n

plt.plot(r, n_F(r))  # doesn't work

def n_BE(r, omega, T):
    #T_c = 0.94*hbar*omega * N**(1/3)
    T_c = 0.28 * 1e-6
    #print(T_c)
    N_0 = N*(1-(T/T_c)**3)
    d = np.sqrt(hbar/(m*omega))
    n_BE = N_0/(np.pi**1.5 * d**3) * np.exp(-r**2 / (d**2))
    return n_BE

plt.figure()
plt.plot(r, n_BE(r, omega, T))  # need to find correct T_c


    


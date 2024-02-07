# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:19:52 2023

@author: kousi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

e_0 = constants.epsilon_0
hbar = constants.hbar
c = constants.c

def susceptibility(Omega_p, Omega_c, Delta_c, Gamma_e, Delta_p, Gamma_r, gamma_p, gamma_c, n, d_ge):
    '''
    Parameters
    ----------
    Omega_p : Rabii frequency of probe beam
    Omega_c : Rabii frequency of couplin beam
    Delta_c : Detuning of coupling beam
    Gamma_e : Spontaneous emission rate of excited state
    Delta_p : Detuning of probe beam (array usually in terms of Gamma_e)
    Gamma_r : Spontaneous emission rate of Rydberg state
    gamma_p : natural linewidth of probe beam
    gamma_c : natural linewidth of coupling beam
    n : number density of medium
    d_ge : Transition dipole moment ground-excited state (in terms of e*a_0)

    Returns
    -------
    (real part of susceptibility, imaginary part of susceptibility)
    '''
    gamma_ge = Gamma_e/2 + gamma_p  # dephasing rate ground - excited
    gamma_gr = Gamma_r/2 + gamma_p + gamma_c  # dephasing rate ground - Rydberg
    
    rho_eg = ((Omega_p/2)*1j)/(gamma_ge - Delta_p*1j + ((Omega_c**2/4)/(gamma_gr-(Delta_p+Delta_c)*1j)))
    
    
    A = (2*n* d_ge**2)/(e_0*hbar*Omega_p)
    
    susceptibility = A * rho_eg
    
    return (np.real(susceptibility), np.imag(susceptibility))

Gamma_e = 6.07 * 1e6 * 2*np.pi
Omega_p = 0.05 * Gamma_e
Omega_c = 0.5 * Gamma_e # 1/3 - 1/2 of Gamma_e is good
Delta_c = 0.0001 * 1e6

Delta_p = np.linspace(-2, 2, 100000) * Gamma_e
Gamma_r = 2.16 * 1e3 * 2*np.pi
gamma_p = 0*10 * 1e3
gamma_c = 0*10 * 1e3

n = 3*1e18  # 10^13 cm^-3 density
d_ge = 5.2 * constants.e * constants.physical_constants['Bohr radius'][0]
wavelength = 780.2 * 1e-9
k_p = 2*np.pi/wavelength

g = (d_ge**2 * k_p)/(2*e_0*hbar)
l=5*1e-6
N = n * l**3  # number of atoms
strength = (n*g*l)/Gamma_e
sigma = (3*wavelength**2)/(2*np.pi)

print(str(strength)+ ' interaction strength')


def plot_susceptibility(Delta_p, susceptibility):
    '''
    Parameters
    ----------
    Delta_p : array of probe Detuning.
    susceptibility : array of (real, imaginary)

    Returns
    -------
    Plots real and imaginary part of the susceptibility
    '''
    maxm = np.max(susceptibility[1])
    plt.plot(Delta_p/Gamma_e, susceptibility[1]/maxm, label = 'Imaginary')
    plt.plot(Delta_p/Gamma_e, susceptibility[0]/maxm, label = 'Real', color='r')
    xtext = r'$\Delta_p/\Gamma_e$'
    ytext = r'$\chi/\tilde{\chi}$'
    plt.xlabel(xtext, fontsize=25)
    plt.ylabel(ytext, fontsize=25)
    #plt.title('Complex susceptibility as a function of probe Detuning', fontsize=20)
    plt.legend(fontsize=15)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.grid()
    

sus = susceptibility(Omega_p, Omega_c, Delta_c, Gamma_e, Delta_p, Gamma_r, gamma_p, gamma_c, n, d_ge)

sus_resonance = susceptibility(Omega_p, 0, Delta_c, Gamma_e, 0, Gamma_r, gamma_p, gamma_c, n, d_ge)

OD1 = k_p*l*sus_resonance[1]
OD2 = n*l*sigma
OD3 = 6*np.pi*n*l/(k_p**2)


print(str(OD1)+' optical depth using susceptibility. factor of 2 out?')
print(str(OD2)+' optical depth correct definition')
print(str(OD3)+' optical depth different calculation, same result')

plot_susceptibility(Delta_p, sus)

#plt.savefig(r'C:\Users\kousi\OneDrive\Documents\Y4 project\Pictures\EIT plots\1EIT(G36MHz) ' +str(np.round(Omega_p/Gamma_e, decimals=2)) + ' ' + str(np.round(Omega_c/Gamma_e, decimals=2)) +  'normalised.png', bbox_inches='tight') 

def group_velocity(Delta_p, real_susceptibility, wavelength):
    '''
    Parameters
    ----------
    Delta_p : array of probe detuning
    real_susceptibility : array of real part of susceptibility
    wavelength : wavelength of probe beam

    Returns
    -------
    v_g : group velocity at resonance

    '''
    susceptibility_0 = np.interp(0, Delta_p, real_susceptibility)
    grad = np.gradient(real_susceptibility, Delta_p)
    grad_0 = np.interp(0, Delta_p, grad)
    
    
    om_p = 2*np.pi * c / wavelength
    
    v_g = c/(1 + 1/2 * susceptibility_0 + 1/2 * om_p*grad_0)
    return v_g


v_g = group_velocity(Delta_p, sus[0], wavelength)
print('group velicity is '+str(v_g)+' m/s')

r'''
Omega_c = np.linspace(1, 20, 200) * 1e6
v_g_array = np.array([])
for i in range(Omega_c.size):
    sus = susceptibility(Omega_p, Omega_c[i], Delta_c, Gamma_e, Delta_p, Gamma_r, gamma_p, gamma_c, n, d_ge)
    v_g = group_velocity(Delta_p, sus[0], wavelength)
    v_g_array = np.append(v_g_array, v_g)
    
print(r'The lowest group velocity (m/s) is ' + str(np.amin(v_g_array)))
print(r'The lowest point happens at Omega_c/Gamma_e ' + str(Omega_c[np.argmin(v_g_array)]/Gamma_e))

fig2 = plt.figure()
plt.plot(Omega_c/Gamma_e, v_g_array)
xtext = r'$\Omega_c/\Gamma_e$'
ytext = r'$v_g$ (m/s)'
plt.xlabel(xtext, fontsize=25)
plt.ylabel(ytext, fontsize=25)
plt.grid()

plt.savefig(r'C:\Users\kousi\OneDrive\Documents\Y4 project\Pictures\EIT plots\v_g 10Omega_p=Gamma_e.png', bbox_inches='tight')
'''
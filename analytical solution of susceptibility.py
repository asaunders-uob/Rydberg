# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:59:31 2023

@author: kousi
"""

import numpy as np
import matplotlib.pyplot as plt

omega_p = 1/10
omega_c = 1/2
delta_p = np.linspace(-3, 3, 300)
delta_c = 0.00001
gamma_ge = 1
gamma_gr = 0.00001

rho_eg = ((omega_p/2)*1j)/(gamma_ge - delta_p*1j + ((omega_c**2/4)/(gamma_gr-(delta_p+delta_c)*1j)))

plt.plot(delta_p, np.imag(rho_eg))

rho_2level = -omega_p/2 * ((delta_p-gamma_ge*1j)/(omega_p**2 / 2 + gamma_ge**2 + delta_p**2))
plt.plot(delta_p, np.imag(rho_2level))
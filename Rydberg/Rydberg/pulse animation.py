# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:48:14 2023

@author: kousi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

omega = 5
v_g = 0.5

t_0 = 1.2
k_0 = 5
k_1 = 0.1  # = 1/v_g
k_2 = 0.2


# figure set up
fig = plt.figure()
ax = plt.axes(xlim=(-10, 50), ylim=(-1.5, 1.5))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function
def animate(t):
    z = np.linspace(-8, 50, 1000)
    sigma = (t_0**2 + ((k_2*z)/t_0)**2)**0.5  # dispersion
    A = (t_0**4 / (t_0**4 + (k_2*z)**2))**0.25  # amplitude
    h = np.exp(-(t-z/v_g)**2 / (2*sigma**2))  # Gaussian envelope
    w = np.exp((omega * t - k_0 * z)*1j)  # wave equation
    f = A * h * w
    line.set_data(z, f)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=30, blit=True)

plt.show()
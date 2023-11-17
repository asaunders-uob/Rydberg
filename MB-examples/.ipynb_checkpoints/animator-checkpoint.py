# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:21:50 2023

@author: A_SAU
"""

from matplotlib import animation
from maxwellbloch import mb_solve

# mb_solve_00 = mb_solve.MBSolve().fro

%time Omegas_zt, states_zt = mb_solve_00.mbsolve(recalc=False)

fig = plt.figure()
ax = plt.axes(xlim=(-10, 50), ylim=(-1.5, 1.5))
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(t):
    return mb_solve_00.Omegas_zt[0][t],

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=30, blit=True)

plt.show()
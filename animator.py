# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:21:50 2023

@author: A_SAU
"""

from matplotlib import animation
from maxwellbloch import mb_solve
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TKAgg')

mb_solve_json = """
{
  "atom": {
    "decays": [
      { "channels": [[0,1], [1,2]],
        "rate": 0.0
      }
    ],
    "energies": [],
    "fields": [
      {
        "coupled_levels": [[0, 1]],
        "detuning": 0.0,
        "detuning_positive": true,
        "label": "probe",
        "rabi_freq": 1.0e-3,
        "rabi_freq_t_args":
          {
             "ampl": 1.0,
             "centre": 0.0,
             "fwhm": 1.0
          },
        "rabi_freq_t_func": "gaussian"
      },
      {
        "coupled_levels": [[1, 2]],
        "detuning": 0.0,
        "detuning_positive": true,
        "label": "coupling",
        "rabi_freq": 5.0,
        "rabi_freq_t_args":
        {
          "ampl": 1.0,
          "fwhm": 0.2,
          "off": 4.0,
          "on": 6.0
          },
        "rabi_freq_t_func": "ramp_offon"
      }
    ],
    "num_states": 3
  },
  "t_min": -2.0,
  "t_max": 12.0,
  "t_steps": 140,
  "z_min": -0.2,
  "z_max": 1.2,
  "z_steps": 140,
  "z_steps_inner": 50,
  "num_density_z_func": "gaussian",
  "num_density_z_args": {
    "ampl": 1.0,
    "fwhm": 0.5,
    "centre": 0.5
  },
  "interaction_strengths": [1.0e3, 1.0e3],
  "velocity_classes": null,
  "method": "mesolve",
  "opts": {},
  "savefile": "qu/mb-solve-lambda-weak-pulse-cloud-atoms-some-coupling-store"
}
"""

mb_solve_00 = mb_solve.MBSolve().from_json_str(mb_solve_json)


Omegas_zt, states_zt = mb_solve_00.mbsolve(recalc=False)


fig = plt.figure()
ax = plt.axes()
line = ax.plot(mb_solve_00.zlist, mb_solve_00.Omegas_zt[0][0], color='k', lw=2)[0]

def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_ydata(mb_solve_00.Omegas_zt[0][i])


anim = animation.FuncAnimation(fig, animate,
                               frames=140, interval=30, blit=False)
anim.save("motion.gif")
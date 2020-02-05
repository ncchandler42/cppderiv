#!/usr/bin/env python3

from time import time

import numpy as np
import matplotlib.pyplot as plt

import cppderiv

##############################################################
# diff_eq:
# description of the problem
#
# row 0 -> 0th-order deriv
# row 1 -> 1st-order deriv
# row 2 -> 2nd-order deriv
# ...
# row n -> nth-order deriv
#
# each column represents a dimension if the terms are vectors
###############################################################

# basic projectile in 3 dims,
# dr/dt = v
# dv_y/dt = -g, dv_x/dt = dv_z/dt = 0

diff_eq = [
	["vals[1, 0]", "vals[1,1]", "vals[1][2]"], 
	["0", "-g", "0"],  
]

# any constant variables used in the diff_eq expressions
consts = {"g": 9.80, "pi": np.pi}

# Timestep
dt = 1e-3

# max time to run in seconds
timeout = 10.0

# create the object: an object describes a single set of ODEs
n_deriv = cppderiv.NDeriv(diff_eq, consts, dt, timeout)

# vals[0,1] = r[1] = y, stop when the projectile hits the ground or 30 seconds have passed
vals_initial = np.array([
	[1.0, 1.0, 0.0],   # 0th-order term, r
	[10.0, 10.0, 0.0], # 1st-order term, v
])

stop_condition = "vals[0, 1] < 0 or t > 30"

start = time()

# euler, rk2, rk4, leapfrog
n_deriv.rk2(vals_initial, stop_condition)

print("Elapsed: {:1.2f} s".format(time() - start))

data = np.array(n_deriv.get_plot_data())

plt.plot(data[:,1], data[:,2])
plt.title("Projectile Motion")
plt.xlabel("$x$ (m)")
plt.ylabel("$y$ (m)")
plt.grid()
plt.show()
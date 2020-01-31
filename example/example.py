#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from time import time

import cppderiv

g = -9.8

def deriv_func(vals_n):
	return np.array([
		vals_n[1], # dr/dt = v
		vals_n[2], # dv/dt = a
		[0,0]      # da/dt = 0			
	])

def stop_condition(vals_n):
	return vals_n[0, 1] < 0 # stop when r[1] < 0, or y < 0
	
timeout = 10 # seconds

n_deriv = cppderiv.NDeriv(deriv_func, stop_condition, timeout)

################################################################

vals_initial = np.array([
	[1.0, 1.0],   # 0th-order term, r
	[10.0, 10.0], # 1st-order term, v
	[0.0, g]      # 2nd-order term, a
])

dt = 1e-4

start = time()

# euler, rk2, rk4, leapfrog
n_deriv.rk2(vals_initial, dt)

print("Elapsed: {:1.2f} s".format(time() - start))

data = np.array(n_deriv.get_plot_data())

plt.plot(data[:,0], data[:,1])
plt.title("Projectile Motion")
plt.xlabel("$x$ (m)")
plt.ylabel("$y$ (m)")
plt.grid()
plt.show()
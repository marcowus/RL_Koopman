#File for constants
import numpy as np

V       = 20.0
k       = 300.0 
N       = 5.0
Tf      = 0.3947
alpha   = 1.95e-04
Tc      = 0.3816


lb_pinn = np.array([0.0,      0.8,     0.0,     0.11,  0.8*0.7293])
ub_pinn = np.array([1.0,      1.2,     700,     0.16,  1.2*0.7293])

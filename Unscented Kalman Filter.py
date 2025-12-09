import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Function that defines the two-body equations of motion
def two_body(t,s):
    mu = 3.986004418e5 #km^3/s^2
    r = s[:3]
    v = s[3:6]
    
    rdot = v
    vdot = -(mu*r)/(np.linalg.norm(r)**3)
    
    sdot = np.concatenate((rdot,vdot))
    
    return sdot


# Initial Conditions
mean = np.array([42241.1,0,0,0,3.07186,0])
sigma_r = (1/1000)*np.linalg.norm(mean[:3])
sigma_v = (1/1000)*np.linalg.norm(mean[3:6])
covR = (sigma_r**2)*np.eye(3)
covV = (sigma_v**2)*np.eye(3)
covariance = np.block([[covR, np.zeros((3,3))],[np.zeros((3,3)), covV]])
n = 6 # System Dimension
Q = 1e-3*np.eye(6) # Process Noise

# Determining the sigma points
#### Do the sigma points change during the filtering or are they determined once and used continuously?
sigma = np.zeros((6,12))
P_half = np.linalg.cholesky(covariance)
e = np.block([np.identity(len(mean)),-np.identity(len(mean))])
for i in range(2*len(mean)):
    sigma[:,i] = mean + np.sqrt(n)*P_half@e[:,i]

# Initializing variables for the propagation
t_final = 2*86400
dt = 10
s_new = np.zeros((6,12))
t = 0
s0 = sigma
propagated_means = np.zeros(int(t_final / dt))
propagated_covariances = np.zeros((n, n, int(t_final / dt)))

# Unscented Kalman Filter
for i in range(0,t_final):

    # Propagating the sigma points
    for i in range(2*len(mean)):
        sol = solve_ivp(two_body,[0,dt],s0[:,i],method='RK45',rtol=1e-8, atol=1e-8)
        s_new[:,i] = sol.y[-1,:]
    
    s0 = s_new

    # Compute the mean and covariance of the propagated sigma points at each time step
    propagated_means[i] = np.mean(sol.y[-1,:], axis=0)

    for j in range(len(sol.t)):
        diffs =  - propagated_means[i]
        propagated_covariances[:,:,i] = (1 / (2 * n)) * sum(np.outer(d, d) for d in diffs)


    # Performing the Kalman Update






    t += dt

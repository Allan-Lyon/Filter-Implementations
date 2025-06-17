import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def odedynamics(t,s0):
    r0 = s0[0]
    v0 = s0[1]
    rdot = v0
    vdot = -r0 - 0.05*v0
    
    sdot = np.array([rdot,vdot])
    
    return sdot

def undamped_dynamics(t, s):
    r = s[0]
    v = s[1]
    return np.array([v, -r])


# Given Information #
t_measure = 0.1
sigma_v = 0.1
r0 = 1
v0 = 0
P0 = np.diag([0.1**2, 0.001**2])
tf = 6*np.pi
dt = 0.05
tspan = np.linspace(0,tf,int(tf/dt))

# Initializing storage matrices #
num_meas = int(tf / t_measure)
r_measure = np.full(num_meas, np.nan)  # using NaN for skipped steps
s0 = np.array([r0,v0])

# True Trajectory #
true_s = solve_ivp(odedynamics,[0,tf],s0,method='RK45',rtol=1e-8, atol=1e-8, dense_output = True)

# Extracting True Position and Velocity #
t_true = true_s.t
r = true_s.y[0,:]
v = true_s.y[1,:]


# Generating Synthetic Measurement #
for i in range(0, num_meas, 2):
    t_i = i * t_measure
    mean = true_s.sol(t_i)[0]
    r_measure[i] = np.random.normal(mean, sigma_v)

    
# Kalman Filter #

# Initializing Variables #
H = np.array([[1, 0]])  
P = np.zeros([len(tspan),2,2])
P[0,:,:] = P0
S = H @ P0 @ H.T + sigma_v**2
K = (P0 @ H.T) * (1/S)
s = np.zeros([len(tspan),2])
s[0,:] = np.array([r0,v0])
Q = np.diag([1e-5,1e-4])
#Q = np.array([0,0])

for i in range(1,len(tspan)):
    # Propagate the mean state

    # Calculate the state transition matrix
    STM = np.array([[np.cos(dt), np.sin(dt)],[-np.sin(dt), np.cos(dt)]])
    s[i,:] = STM @ s[i-1,:]
    
    # Propagate the covariance
    P[i,:,:] = STM @ P[i-1,:,:] @ STM.T + Q
    
    # If a measurement occurs, perform a Kalman update
    if i % int(t_measure/dt) == 0:
        meas_idx = int(i * dt / t_measure)
        if meas_idx < len(r_measure) and not np.isnan(r_measure[meas_idx]):
            S = H @ P[i,:,:] @ H.T + sigma_v**2 
            K = (P[i,:,:] @ H.T) * (1/S)
            s[i,:] = s[i,:] + K @ (np.array([r_measure[meas_idx]]) - H @ s[i,:])
            P[i,:,:] = (np.eye(2) - K @ H) @ P[i,:,:] @ (np.eye(2) - K @ H).T + K @ K.T * sigma_v**2


# Extract 3-sigma bounds from covariance matrix
sigma_r = 3 * np.sqrt(P[:,0,0])  # 3-sigma for position
sigma_v_KF = 3 * np.sqrt(P[:,1,1])

# Extracting the Kalman Filter states #
r_Kalman = s[:,0]
v_Kalman = s[:,1]

# Plotting the Position
plt.plot(t_true,r,label="True")
plt.plot(tspan,r_Kalman,label="Kalman")
plt.fill_between(tspan, r_Kalman - sigma_r, r_Kalman + sigma_r, color='gray', alpha=0.3, label="3-sigma envelope")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.legend(loc='upper right')
plt.show()

# Plotting the Velocity
plt.plot(t_true,v,label="True")
plt.plot(tspan,v_Kalman,label="Kalman")
plt.fill_between(tspan, v_Kalman - sigma_v_KF, v_Kalman + sigma_v_KF, color='gray', alpha=0.3, label="3-sigma Envelope")
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
plt.legend(loc='upper right')
plt.show()

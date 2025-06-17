import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def truth_dynamics(t,s,mu):
    r = s[:3]
    v = s[3:6]
    
    rdot = v
    vdot = -(mu*r)/(np.linalg.norm(r)**3) - 1e-9*(v/np.linalg.norm(v))
    
    sdot = np.concatenate((rdot,vdot))
    
    return sdot 

def dynamics_with_stm(t, y, mu):
    x = y[:6]
    Phi = y[6:].reshape((6, 6))

    r = x[:3]
    v = x[3:]

    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    
    # Two-body acceleration
    a = -(mu * r) / r_norm**3

    # Dynamics
    x_dot = np.zeros(6)
    x_dot[:3] = v
    x_dot[3:] = a

    # Compute Jacobian A
    I3 = np.eye(3)
    O3 = np.zeros((3,3))
    A = np.block([
        [O3,        I3],
        [-mu*(np.eye(3)/r_norm**3 - 3*np.outer(r,r)/r_norm**5), O3]
    ])

    # STM derivative
    Phi_dot = A @ Phi

    return np.concatenate((x_dot, Phi_dot.flatten()))

# Given Values #
mu = 398600.4418 # km^3/s^2
z = 400 # km
earth_radius = 6371 # km

# Time setup
t_days = 30
t_final = t_days * 24 * 3600  # seconds
dt = 1*60  # step size in seconds
t_span = (0, t_final)
t_eval = np.linspace(0, t_final, int(t_final/dt))


# Truth Simulation
r0 = [earth_radius+z,0,0]
v0 = [0,np.sqrt(mu / np.linalg.norm(r0)),0]
s0 = np.concatenate((r0,v0))
s_true = solve_ivp(truth_dynamics,t_span,s0,t_eval=t_eval,rtol=1e-6, atol=1e-8,args=(mu,))

# Plotting the distance from the center of earth
radius = np.zeros((len(s_true.t),1))
for i in range(0,len(s_true.t)):
    radius[i] = np.sqrt(s_true.y[0, i]**2 + s_true.y[1, i]**2 + s_true.y[2, i]**2)

plt.plot(s_true.t/3600/24,radius)
plt.xlabel("Time (days)")
plt.ylabel("Distance (km)")
plt.title("Distance From Center of the Earth")
plt.show()

plt.plot(s_true.y[0], s_true.y[1])
plt.axis("equal")
plt.title("Orbit Trajectory")
plt.xlabel("X (km)")
plt.ylabel("Y (km)")
plt.show()

# Simulating Measurements
sigma = 1e-3
dt_meas = 1*3600
t_meas = np.arange(0,t_final+dt_meas,dt_meas)
H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]])
R = np.diag([sigma**2,sigma**2,sigma**2])
r_measure = np.zeros((len(t_meas),3))
for i in range(0,len(t_meas)):
    nu = np.random.multivariate_normal(np.zeros(3), R, 1).reshape(3,1)
    mean = solve_ivp(truth_dynamics,[0,dt_meas*i],s0,method='RK45',rtol=1e-6,atol=1e-8,args=(mu,))
    r_measure[i,:] = mean.y[:3,-1] + nu.flatten()
 
# Extended Kalman Filter
#Q = np.zeros((6,6)) # Process Noise
Q = 1e-11*np.eye(6)
s = np.zeros([len(t_eval),6])
P = np.zeros([len(t_eval),6,6])
P[0,:,:] = np.block([[np.eye(3),np.zeros((3,3))],[np.zeros((3,3)),1e-8*np.eye(3)]])
s[0,:] = np.concatenate([r0,v0])
for i in range(1,len(t_eval)):
    # Propagate the mean state and calculate the state transition matrix
    Phi0 = np.eye(6).flatten()
    aug_state0 = np.concatenate([s[i-1,:], Phi0])

    # Integrate the augmented state
    aug_sol = solve_ivp(dynamics_with_stm, [0, dt], aug_state0, args=(mu,), rtol=1e-6, atol=1e-8)
    s[i,:] = aug_sol.y[:6,-1]
    STM = aug_sol.y[6:,-1].reshape((6,6))

    # Propagate the covariance
    P[i,:,:] = STM @ P[i-1,:,:] @ STM.T + Q
    
    # If a measurement occurs, perform a Kalman update
    if i % int(dt_meas/dt) == 0:
        meas_idx = int(i * dt / dt_meas)
        if meas_idx < len(r_measure) and np.all(np.isfinite(r_measure[meas_idx])):
            S = H @ P[i,:,:] @ H.T + R
            K = P[i,:,:] @ H.T @ np.linalg.inv(S)
            s[i,:] = s[i,:] + K @ (r_measure[meas_idx] - H @ s[i,:])
            P[i,:,:] = (np.eye(6) - K @ H) @ P[i,:,:] @ (np.eye(6) - K @ H).T + K @ R @ K.T


# Plotting the results

skip_steps = int(2 * 3600 / dt)  # 2 hours worth of steps

# Extract magnitude of position and velocity
pos_mag = np.linalg.norm(s[:, :3], axis=1)
vel_mag = np.linalg.norm(s[:, 3:], axis=1)

# True state magnitudes
true_pos_mag = np.linalg.norm(s_true.y[:3].T, axis=1)
true_vel_mag = np.linalg.norm(s_true.y[3:].T, axis=1)

# Extract 3-sigma bounds from covariance matrix
sigma_pos = np.sqrt(np.array([np.trace(P[i, :3, :3])/3 for i in range(len(P))]))
sigma_vel = np.sqrt(np.array([np.trace(P[i, 3:, 3:])/3 for i in range(len(P))]))

# Time vector in days
t_days_vec = t_eval / 86400

# Plotting Position Magnitude and 3-Sigma Bounds
plt.figure()
plt.plot(t_days_vec, pos_mag, label='Estimated Position Magnitude')
plt.plot(t_days_vec, true_pos_mag, '--', label='True Position Magnitude', linewidth=1.5)
plt.fill_between(t_days_vec, pos_mag - 3*sigma_pos, pos_mag + 3*sigma_pos,
                 color='gray', alpha=0.3, label='3σ Envelope')
plt.xlabel("Time (days)")
plt.ylabel("Position Magnitude (km)")
plt.title("Position Magnitude vs Truth and 3σ Bounds")
plt.legend()
plt.show()

plt.figure()
plt.plot(t_days_vec, pos_mag, label='Estimated Position Magnitude')
plt.plot(t_days_vec, true_pos_mag, '--', label='True Position Magnitude', linewidth=1.5)
plt.fill_between(t_days_vec[skip_steps:], 
                 pos_mag[skip_steps:] - 3 * sigma_pos[skip_steps:], 
                 pos_mag[skip_steps:] + 3 * sigma_pos[skip_steps:], 
                 color='gray', alpha=0.3, label='3σ Envelope')
plt.xlabel("Time (days)")
plt.ylabel("Position Magnitude (km)")
plt.title("Position Magnitude vs Truth and 3σ Bounds")
plt.legend()
plt.show()

plt.figure()
plt.plot(t_days_vec, vel_mag, label='Estimated Velocity Magnitude')
plt.plot(t_days_vec, true_vel_mag, '--', label='True Velocity Magnitude', linewidth=1.5)
plt.fill_between(t_days_vec, vel_mag - 3*sigma_vel, vel_mag + 3*sigma_vel,
                 color='gray', alpha=0.3, label='3σ Envelope')
plt.xlabel("Time (days)")
plt.ylabel("Velocity Magnitude (km/s)")
plt.title("Velocity Magnitude vs Truth and 3σ Bounds")
plt.legend()
plt.show()

plt.figure()
plt.plot(t_days_vec, vel_mag, label='Estimated Velocity Magnitude')
plt.plot(t_days_vec, true_vel_mag, '--', label='True Velocity Magnitude', linewidth=1.5)
plt.fill_between(t_days_vec[skip_steps:], 
                 vel_mag[skip_steps:] - 3 * sigma_vel[skip_steps:], 
                 vel_mag[skip_steps:] + 3 * sigma_vel[skip_steps:], 
                 color='gray', alpha=0.3, label='3σ Envelope')
plt.xlabel("Time (days)")
plt.ylabel("Velocity Magnitude (km/s)")
plt.title("Velocity Magnitude vs Truth and 3σ Bounds")
plt.legend()
plt.show()

# Position: Check how often the true position magnitude is within the 3σ bounds
within_3sigma_pos = np.abs(true_pos_mag - pos_mag) <= 3 * sigma_pos
pos_percentage = np.sum(within_3sigma_pos) / len(true_pos_mag) * 100

# Velocity: Check how often the true velocity magnitude is within the 3σ bounds
within_3sigma_vel = np.abs(true_vel_mag - vel_mag) <= 3 * sigma_vel
vel_percentage = np.sum(within_3sigma_vel) / len(true_vel_mag) * 100

# Print results
print(f"Percentage of True Position values within 3σ bounds: {pos_percentage:.2f}%")
print(f"Percentage of True Velocity values within 3σ bounds: {vel_percentage:.2f}%")
